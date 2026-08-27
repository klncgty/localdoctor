"""A tiny server that mimics Ollama's response format.

Tests must not require a GPU or a real Ollama (spec 11). The scenario is chosen
with the `x-fake-scenario` header, which passes through the proxy unchanged.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODEL_CONTEXT_LENGTH = 262144
DEFAULT_NUM_CTX = 4096
LONG_THINKING = "Let me think about this. " * 400
NORMAL_CONTENT = "This is a normal answer from {model}."

app = FastAPI()

# Raw requests that reached the server, so tests can assert on them.
seen_requests: list[dict[str, Any]] = []


def scenario_of(request: Request) -> str:
    return request.headers.get("x-fake-scenario", "normal")


def prompt_chars(body: dict[str, Any]) -> int:
    """Rough size of the text this request carries."""
    total = len(str(body.get("prompt") or "")) + len(str(body.get("system") or ""))
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += len(content)
    return total


def plan_for(scenario: str, body: dict[str, Any]) -> dict[str, Any]:
    """Produce the measured values of a response for the given scenario."""
    num_ctx = int((body.get("options") or {}).get("num_ctx") or DEFAULT_NUM_CTX)
    num_predict = int((body.get("options") or {}).get("num_predict") or 128)

    # A real server reports what it actually read. ~4 chars per token is the
    # usual BPE ratio, comfortably above the lower bound LocalDoctor assumes.
    realistic = max(8, prompt_chars(body) // 4)

    plan = {
        "content": NORMAL_CONTENT.format(model=body.get("model", "fake:latest")),
        "thinking": "",
        "done_reason": "stop",
        "prompt_eval_count": min(realistic, num_ctx),
        "eval_count": 9,
        "num_ctx": num_ctx,
    }

    if scenario == "drip":
        plan["content"] = "chunk " * 12
    if scenario == "truncated":
        # Pinned to the window: S1 fires.
        plan["prompt_eval_count"] = num_ctx
    elif scenario == "context_shift":
        # What real Ollama does: --context-shift drops part of an oversized
        # prompt, so the count lands well below num_ctx and S1 never fires.
        plan["prompt_eval_count"] = max(8, num_ctx // 2)
        plan["done_reason"] = "length"
    elif scenario == "near_limit_healthy":
        # Near the limit but not pinned: no signal may fire.
        plan["prompt_eval_count"] = max(1, num_ctx - 50)
    elif scenario == "cached_prompt":
        # A KV-cache hit: the server reports only the newly evaluated tokens,
        # so the count lands below the lower bound of what was sent. This is
        # exactly why S3 on its own is a heuristic, not evidence.
        plan["prompt_eval_count"] = 8
    elif scenario == "empty":
        plan["content"] = "   \n  "
        plan["eval_count"] = 7
    elif scenario == "reasoning_starved":
        plan["thinking"] = LONG_THINKING
        plan["content"] = ""
        plan["done_reason"] = "length"
        plan["eval_count"] = num_predict
    elif scenario == "reasoning_think_tags":
        plan["content"] = f"<think>{LONG_THINKING}</think>"
        plan["done_reason"] = "length"
        plan["eval_count"] = num_predict
    elif scenario == "reasoning_healthy":
        plan["thinking"] = "A short thought."
        plan["content"] = NORMAL_CONTENT
    elif scenario == "truncated_and_empty":
        # R001 + R003 together: exercises suppression.
        plan["prompt_eval_count"] = num_ctx
        plan["content"] = ""
        plan["eval_count"] = 7
    return plan


def ollama_final(model: str, plan: dict[str, Any], kind: str) -> dict[str, Any]:
    base = {
        "model": model,
        "created_at": "2026-01-01T00:00:00.000000Z",
        "done": True,
        "done_reason": plan["done_reason"],
        "total_duration": 1_000_000,
        "load_duration": 10_000,
        "prompt_eval_count": plan["prompt_eval_count"],
        "prompt_eval_duration": 500_000,
        "eval_count": plan["eval_count"],
        "eval_duration": 400_000,
    }
    if kind == "chat":
        base["message"] = {"role": "assistant", "content": "", "thinking": ""}
    else:
        base["response"] = ""
    return base


DRIP_DELAY = 0.05


async def ollama_stream(model: str, plan: dict[str, Any], kind: str, slow: bool, drip: float = 0.0):
    """NDJSON stream. Chunks go out one by one, without delay."""
    pieces = [plan["content"][i : i + 8] for i in range(0, len(plan["content"]), 8)] or [""]
    if slow:
        await asyncio.sleep(0.3)
    if plan["thinking"]:
        chunk = {"model": model, "created_at": "t", "done": False}
        if kind == "chat":
            chunk["message"] = {"role": "assistant", "content": "", "thinking": plan["thinking"]}
        else:
            chunk["response"] = ""
            chunk["thinking"] = plan["thinking"]
        yield (json.dumps(chunk) + "\n").encode()
    for piece in pieces:
        chunk = {"model": model, "created_at": "t", "done": False}
        if kind == "chat":
            chunk["message"] = {"role": "assistant", "content": piece, "thinking": ""}
        else:
            chunk["response"] = piece
        yield (json.dumps(chunk) + "\n").encode()
        await asyncio.sleep(drip)
    yield (json.dumps(ollama_final(model, plan, kind)) + "\n").encode()


async def openai_stream(model: str, plan: dict[str, Any], slow: bool):
    """SSE stream."""
    if slow:
        await asyncio.sleep(0.3)
    text = plan["thinking"] + plan["content"]
    pieces = [text[i : i + 8] for i in range(0, len(text), 8)] or [""]
    for piece in pieces:
        chunk = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0)
    finish = "length" if plan["done_reason"] == "length" else "stop"
    tail = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": plan["prompt_eval_count"],
            "completion_tokens": plan["eval_count"],
            "total_tokens": plan["prompt_eval_count"] + plan["eval_count"],
        },
    }
    yield f"data: {json.dumps(tail)}\n\n".encode()
    yield b"data: [DONE]\n\n"


last_raw_body: dict[str, bytes] = {}


async def read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    last_raw_body[request.url.path] = raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


@app.post("/api/chat")
@app.post("/api/generate")
async def native(request: Request):
    body = await read_json(request)
    seen_requests.append({"path": request.url.path, "body": body, "headers": dict(request.headers)})
    scenario = scenario_of(request)
    if scenario == "error":
        return JSONResponse({"error": "model 'yok' not found"}, status_code=404)
    model = body.get("model", "fake:latest")
    kind = "chat" if request.url.path.endswith("/chat") else "generate"
    plan = plan_for(scenario, body)
    slow = scenario == "slow"
    if body.get("stream", True):
        drip = DRIP_DELAY if scenario == "drip" else 0.0
        return StreamingResponse(
            ollama_stream(model, plan, kind, slow, drip), media_type="application/x-ndjson"
        )
    if slow:
        await asyncio.sleep(0.3)
    final = ollama_final(model, plan, kind)
    if kind == "chat":
        final["message"] = {
            "role": "assistant",
            "content": plan["content"],
            "thinking": plan["thinking"],
        }
    else:
        final["response"] = plan["content"]
        final["thinking"] = plan["thinking"]
    return JSONResponse(final)


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def openai_compat(request: Request):
    body = await read_json(request)
    seen_requests.append({"path": request.url.path, "body": body, "headers": dict(request.headers)})
    scenario = scenario_of(request)
    if scenario == "error":
        return JSONResponse({"error": {"message": "not found"}}, status_code=404)
    model = body.get("model", "fake:latest")
    plan = plan_for(scenario, body)
    if body.get("stream", False):
        return StreamingResponse(
            openai_stream(model, plan, scenario == "slow"), media_type="text/event-stream"
        )
    finish = "length" if plan["done_reason"] == "length" else "stop"
    return JSONResponse(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": plan["thinking"] + plan["content"],
                    },
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": plan["prompt_eval_count"],
                "completion_tokens": plan["eval_count"],
                "total_tokens": plan["prompt_eval_count"] + plan["eval_count"],
            },
        }
    )


@app.post("/api/embed")
async def embed(request: Request):
    body = await read_json(request)
    seen_requests.append({"path": request.url.path, "body": body, "headers": dict(request.headers)})
    return JSONResponse(
        {"model": body.get("model", "fake:latest"), "embeddings": [[0.1, 0.2]], "prompt_eval_count": 4}
    )


@app.post("/api/show")
async def show(request: Request):
    body = await read_json(request)
    model = body.get("model") or body.get("name") or "fake:latest"
    params = "top_k                          20\ntop_p                          0.95"
    if "smallctx" in model:
        params += "\nnum_ctx                        512"
    return JSONResponse(
        {
            "modelfile": "FROM fake",
            "parameters": params,
            "template": "{{ .Prompt }}",
            "details": {"family": "fake", "parameter_size": "9.0B", "quantization_level": "Q6_K"},
            "model_info": {
                "general.architecture": "fake",
                "fake.context_length": MODEL_CONTEXT_LENGTH,
            },
            "capabilities": ["completion", "thinking"],
            "modified_at": "2026-01-01T00:00:00.000000Z",
        }
    )


@app.get("/api/tags")
async def tags():
    return JSONResponse(
        {"models": [{"name": "fake:latest", "digest": "abc123", "details": {"family": "fake"}}]}
    )


@app.get("/api/version")
async def version():
    return JSONResponse({"version": "0.0.0-fake"})


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def catchall(path: str, request: Request):
    seen_requests.append({"path": "/" + path, "body": None, "headers": dict(request.headers)})
    return JSONResponse({"ok": True, "path": "/" + path})
