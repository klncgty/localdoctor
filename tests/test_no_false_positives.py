"""Spec 11, test 5 — the test that matters most.

A diagnostic tool that cries wolf is worth less than no tool at all. Healthy
traffic must leave the terminal completely clean, while still being recorded.
"""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import CHAT, chat_body

# At or above a quarter of the model's 262,144 context, so R002 stays quiet too
# and the screen must be perfectly empty.
ROOMY = 131_072

HEALTHY_CALLS = [
    # (label, scenario, body)
    ("short request", "normal", chat_body(options={"num_ctx": ROOMY})),
    (
        "long but fitting",
        "normal",
        chat_body(
            messages=[{"role": "user", "content": "x" * 40_000}], options={"num_ctx": ROOMY}
        ),
    ),
    (
        "near the limit, not truncated",
        "near_limit_healthy",
        chat_body(options={"num_ctx": ROOMY}),
    ),
    (
        "reasoning model that answers",
        "reasoning_healthy",
        chat_body(options={"num_ctx": ROOMY, "num_predict": 512}),
    ),
    ("streaming", "normal", chat_body(stream=True, options={"num_ctx": ROOMY})),
    (
        "streaming near the limit",
        "near_limit_healthy",
        chat_body(stream=True, options={"num_ctx": ROOMY}),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label, scenario, body", HEALTHY_CALLS, ids=[c[0] for c in HEALTHY_CALLS])
async def test_healthy_request_prints_nothing(stack, label, scenario, body):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT, json=body, headers={"x-fake-scenario": scenario}
        )
    assert response.status_code == 200
    await stack.settle()

    assert stack.printed() == "", f"false positive on a healthy request: {label}"


@pytest.mark.asyncio
async def test_all_healthy_calls_in_one_session_stay_silent(stack):
    """The whole set together, sharing one reporter and one model-info cache."""
    async with httpx.AsyncClient(timeout=30) as client:
        for _, scenario, body in HEALTHY_CALLS:
            response = await client.post(
                stack.base_url + CHAT, json=body, headers={"x-fake-scenario": scenario}
            )
            assert response.status_code == 200
            await stack.settle()

    assert stack.printed() == ""
    assert stack.store.query("SELECT * FROM diagnoses") == []


@pytest.mark.asyncio
async def test_openai_compatible_path_is_also_silent(stack):
    async with httpx.AsyncClient(timeout=30) as client:
        for stream in (False, True):
            response = await client.post(
                stack.base_url + "/v1/chat/completions",
                json={
                    "model": "fake:latest",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": stream,
                },
            )
            assert response.status_code == 200
            await stack.settle()

    assert stack.printed() == ""


@pytest.mark.asyncio
async def test_embeddings_are_silent(stack):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + "/api/embed", json={"model": "fake:latest", "input": "hello"}
        )
    assert response.status_code == 200
    await stack.settle()
    assert stack.printed() == ""


@pytest.mark.asyncio
async def test_silence_still_records(stack):
    """Silence means healthy, not unobserved. "Why didn't it warn me?" needs data."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT, json=chat_body(options={"num_ctx": ROOMY})
        )
    assert response.status_code == 200
    await stack.settle()

    assert stack.printed() == ""
    rows = stack.store.query("SELECT * FROM requests")
    assert len(rows) == 1
    row = rows[0]
    assert row["endpoint"] == CHAT
    assert row["status_code"] == 200
    assert row["prompt_tokens"] is not None
    assert row["num_ctx"] == ROOMY
    assert row["num_ctx_source"] == "request"
    assert row["request_body"], "raw request body must be stored for replay"
    assert row["total_ms"] is not None


@pytest.mark.asyncio
async def test_untouched_endpoints_are_not_recorded(stack):
    """Dumb passthrough: do not touch, do not analyse, do not store."""
    async with httpx.AsyncClient(timeout=30) as client:
        assert (await client.get(stack.base_url + "/api/tags")).status_code == 200
        assert (await client.get(stack.base_url + "/api/version")).status_code == 200
        assert (await client.get(stack.base_url + "/anything/else")).status_code == 200
    await stack.settle()

    assert stack.printed() == ""
    assert stack.store.query("SELECT * FROM requests") == []
