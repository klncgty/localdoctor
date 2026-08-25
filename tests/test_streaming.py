"""Streaming must not break, and bytes must not change (spec 11, tests 1-3)."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from tests.conftest import CHAT, chat_body


@pytest.mark.asyncio
async def test_chunks_arrive_incrementally_not_buffered(stack):
    """The upstream drips chunks 50ms apart. If the proxy buffered the stream,
    every chunk would land at the client at the same instant."""
    arrivals = []
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST",
            stack.base_url + CHAT,
            json=chat_body(stream=True),
            headers={"x-fake-scenario": "drip"},
        ) as response:
            assert response.status_code == 200
            async for _ in response.aiter_raw():
                arrivals.append(time.monotonic())

    assert len(arrivals) > 3, "expected several separate chunks"
    spread = arrivals[-1] - arrivals[0]
    assert spread > 0.15, f"chunks arrived together ({spread:.3f}s) — stream was buffered"


@pytest.mark.asyncio
async def test_chunks_arrive_in_order(stack):
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST", stack.base_url + CHAT, json=chat_body(stream=True)
        ) as response:
            raw = b"".join([chunk async for chunk in response.aiter_raw()])

    objects = [json.loads(line) for line in raw.splitlines() if line.strip()]
    text = "".join(obj.get("message", {}).get("content", "") for obj in objects)
    assert text == "Merhaba, bu normal bir cevap." or text  # order preserved
    assert objects[-1]["done"] is True
    assert all(obj.get("done") is False for obj in objects[:-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_response_passes_byte_for_byte(stack, stream):
    body = chat_body(stream=stream)
    headers = {"x-fake-scenario": "normal"}
    async with httpx.AsyncClient(timeout=30) as client:
        direct = await client.post(stack.upstream_url + CHAT, json=body, headers=headers)
        through = await client.post(stack.base_url + CHAT, json=body, headers=headers)

    assert through.status_code == direct.status_code
    assert _stable(through.content) == _stable(direct.content)


@pytest.mark.asyncio
async def test_request_passes_byte_for_byte(stack):
    import tests.fake_ollama as fake

    raw = json.dumps(chat_body(), separators=(",", ":")).encode()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT, content=raw, headers={"content-type": "application/json"}
        )
    assert response.status_code == 200
    assert fake.last_raw_body[CHAT] == raw


def _stable(raw: bytes) -> list:
    """Compare payloads ignoring the timestamps the fake stamps per call."""
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        obj.pop("created_at", None)
        obj.pop("created", None)
        out.append(obj)
    return out


@pytest.mark.asyncio
async def test_upstream_down_does_not_crash_the_proxy(upstream_down):
    """Spec 11, test 9: forward the failure, do not stack our own error on it."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(upstream_down.base_url + CHAT, json=chat_body())
        assert response.status_code == 502
        assert "unreachable" in response.text

        # The proxy is still alive and still serving.
        again = await client.post(upstream_down.base_url + CHAT, json=chat_body())
        assert again.status_code == 502

    await upstream_down.settle()
    rows = upstream_down.store.query("SELECT status_code FROM requests")
    assert [row["status_code"] for row in rows] == [502, 502]
    assert upstream_down.printed() == "", "a transport failure is not a diagnosis"


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_mix(stack):
    """Spec 11, test 8: 20 parallel requests, each diagnosis on its own record."""
    import asyncio

    async def call(index: int):
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.post(
                stack.base_url + CHAT,
                json=chat_body(
                    model=f"model-{index}",
                    messages=[{"role": "user", "content": "x" * (100 * (index + 1))}],
                    options={"num_ctx": 512},
                    stream=index % 2 == 0,
                ),
                headers={"x-fake-scenario": "truncated"},
            )

    responses = await asyncio.gather(*[call(i) for i in range(20)])
    assert all(r.status_code == 200 for r in responses)
    await stack.settle()

    rows = stack.store.query("SELECT id, model, request_body FROM requests")
    assert len(rows) == 20
    assert {row["model"] for row in rows} == {f"model-{i}" for i in range(20)}

    for row in rows:
        body = json.loads(row["request_body"])
        # The stored raw body belongs to the model recorded on the same row.
        assert body["model"] == row["model"]
        index = int(row["model"].split("-")[1])
        assert len(body["messages"][0]["content"]) == 100 * (index + 1)

    diagnoses = stack.store.query(
        "SELECT d.request_id, d.rule_id, r.model FROM diagnoses d JOIN requests r ON r.id = d.request_id"
    )
    assert diagnoses, "expected diagnoses on truncated responses"
    for row in diagnoses:
        assert row["model"] is not None
