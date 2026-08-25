"""Response normalization (spec 4.2).

NO provider abstraction. One normalize function with two implementations:
`_from_ollama_native` and `_from_openai_compat`. llama.cpp, vLLM and LM Studio
already speak OpenAI-compatible; all three go through the second one. If a new
provider is ever needed, add an `elif` here — do not write a plugin registry.
"""

from __future__ import annotations

import gzip
import json
import zlib
from dataclasses import dataclass, field
from typing import Any

from localdoctor.models import Usage

OLLAMA_NATIVE_PATHS = {"/api/chat", "/api/generate", "/api/embed"}
OPENAI_COMPAT_PATHS = {"/v1/chat/completions", "/v1/completions"}
ANALYZED_PATHS = OLLAMA_NATIVE_PATHS | OPENAI_COMPAT_PATHS

# Paths that invalidate a model's cache entry when seen on the proxy (spec 4.5).
CACHE_MUTATING_PATHS = {"/api/pull", "/api/create", "/api/delete", "/api/copy", "/api/push"}


def api_kind(path: str) -> str | None:
    if path in OLLAMA_NATIVE_PATHS:
        return "ollama_native"
    if path in OPENAI_COMPAT_PATHS:
        return "openai_compat"
    return None


def maybe_decompress(raw: bytes, content_encoding: str | None) -> bytes:
    """Decompress the raw body for analysis only. Passthrough is never touched."""
    if not raw or not content_encoding:
        return raw
    enc = content_encoding.lower().strip()
    try:
        if enc == "gzip":
            return gzip.decompress(raw)
        if enc == "deflate":
            return zlib.decompress(raw)
    except Exception:
        return b""
    return raw if enc == "identity" else b""


# --- request side -------------------------------------------------------


def extract_prompt_text(body: dict[str, Any], path: str) -> str:
    """Join every piece of text sent to the model in this request.

    Images (base64) are excluded: they are not text tokens. The result is still
    a SUBSET of what was actually sent, so the lower bound stays valid.
    """
    if not isinstance(body, dict):
        return ""
    parts: list[str] = []

    for key in ("system", "prompt", "suffix", "input", "template"):
        value = body.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(v for v in value if isinstance(v, str))

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # OpenAI multi-part content: [{"type":"text","text":...}, ...]
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
        if isinstance(message.get("thinking"), str):
            parts.append(message["thinking"])
        for call in message.get("tool_calls") or []:
            parts.append(json.dumps(call, ensure_ascii=False))

    # Tool definitions are embedded into the prompt too; they really do cost context.
    if body.get("tools"):
        parts.append(json.dumps(body["tools"], ensure_ascii=False))

    return "\n".join(parts)


def requested_num_ctx(body: dict[str, Any]) -> int | None:
    """Explicit num_ctx from the request. The OpenAI-compatible path has no such field."""
    options = body.get("options")
    if isinstance(options, dict):
        value = options.get("num_ctx")
        if isinstance(value, int) and value > 0:
            return value
    return None


def requested_num_predict(body: dict[str, Any]) -> int | None:
    options = body.get("options")
    if isinstance(options, dict):
        value = options.get("num_predict")
        if isinstance(value, int) and value > 0:
            return value
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


# --- response side ------------------------------------------------------


@dataclass(slots=True)
class ParsedResponse:
    """Measurements and text extracted from a raw response."""

    meta: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    thinking: str = ""
    ok: bool = False


def _iter_json_lines(raw: bytes):
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def _iter_sse_data(raw: bytes):
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            yield json.loads(payload)
        except Exception:
            continue


def parse_response(path: str, raw: bytes, stream: bool) -> ParsedResponse:
    """Turn a raw response body into measurements and text."""
    kind = api_kind(path)
    if not raw or kind is None:
        return ParsedResponse()
    try:
        if kind == "ollama_native":
            return _parse_ollama(raw, stream)
        return _parse_openai(raw, stream)
    except Exception:
        return ParsedResponse()


def _parse_ollama(raw: bytes, stream: bool) -> ParsedResponse:
    out = ParsedResponse()
    objects = list(_iter_json_lines(raw)) if stream else _single_json(raw)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        out.ok = True
        message = obj.get("message")
        if isinstance(message, dict):
            out.content += _as_text(message.get("content"))
            out.thinking += _as_text(message.get("thinking"))
        out.content += _as_text(obj.get("response"))
        out.thinking += _as_text(obj.get("thinking"))
        # Measurements only arrive in the final (done) object; merge them all.
        for key in (
            "model",
            "done",
            "done_reason",
            "prompt_eval_count",
            "eval_count",
            "total_duration",
            "error",
        ):
            if key in obj:
                out.meta[key] = obj[key]
    return out


def _parse_openai(raw: bytes, stream: bool) -> ParsedResponse:
    out = ParsedResponse()
    objects = list(_iter_sse_data(raw)) if stream else _single_json(raw)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        out.ok = True
        if "model" in obj:
            out.meta["model"] = obj["model"]
        if isinstance(obj.get("usage"), dict):
            out.meta["usage"] = obj["usage"]
        if "error" in obj:
            out.meta["error"] = obj["error"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                out.meta["finish_reason"] = choice["finish_reason"]
            for holder_key in ("delta", "message"):
                holder = choice.get(holder_key)
                if isinstance(holder, dict):
                    out.content += _as_text(holder.get("content"))
                    out.thinking += _as_text(holder.get("reasoning_content"))
                    out.thinking += _as_text(holder.get("reasoning"))
            out.content += _as_text(choice.get("text"))
    return out


def _single_json(raw: bytes) -> list[Any]:
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    return [obj] if isinstance(obj, dict) else []


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


# --- normalize ----------------------------------------------------------


def normalize(request_body: dict[str, Any], response_meta: dict[str, Any], path: str) -> Usage:
    """normalize(request_body, response_meta) -> Usage"""
    kind = api_kind(path)
    if kind == "openai_compat":
        usage = _from_openai_compat(response_meta)
    else:
        usage = _from_ollama_native(response_meta)
    if usage.model is None and isinstance(request_body.get("model"), str):
        usage.model = request_body["model"]
    usage.num_ctx_requested = requested_num_ctx(request_body)
    return usage


def _from_ollama_native(meta: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=_as_int(meta.get("prompt_eval_count")),
        completion_tokens=_as_int(meta.get("eval_count")),
        finish_reason=meta.get("done_reason") if isinstance(meta.get("done_reason"), str) else None,
        model=meta.get("model") if isinstance(meta.get("model"), str) else None,
    )


def _from_openai_compat(meta: dict[str, Any]) -> Usage:
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    return Usage(
        prompt_tokens=_as_int(usage.get("prompt_tokens")),
        completion_tokens=_as_int(usage.get("completion_tokens")),
        finish_reason=meta.get("finish_reason") if isinstance(meta.get("finish_reason"), str) else None,
        model=meta.get("model") if isinstance(meta.get("model"), str) else None,
    )


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
