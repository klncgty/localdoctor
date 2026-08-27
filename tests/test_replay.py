"""Phase 3: replay recorded traffic against other models."""

from __future__ import annotations

import json

import httpx
import pytest

from localdoctor.replay import replay
from tests.conftest import CHAT, chat_body

import tests.fake_ollama as fake


async def record_one(stack, **overrides):
    body = chat_body(options={"num_ctx": 131_072}, **overrides)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(stack.base_url + CHAT, json=body)
        assert response.status_code == 200
    await stack.settle()
    return stack.store.query("SELECT * FROM requests ORDER BY ts DESC")[0]


@pytest.mark.asyncio
async def test_replay_sends_to_each_model_and_keeps_a_baseline(stack):
    row = await record_one(stack)
    run = await replay(row, ["model-a", "model-b"], stack.upstream_url)

    assert [r.model for r in run.results] == ["fake:latest", "model-a", "model-b"]
    assert run.baseline is not None and run.baseline.model == "fake:latest"
    assert all(r.error is None for r in run.results)

    # Only the model field changed; everything else was replayed as recorded.
    replayed = [r for r in fake.seen_requests if r["body"].get("model", "").startswith("model-")]
    assert {r["body"]["model"] for r in replayed} == {"model-a", "model-b"}
    original = json.loads(row["request_body"])
    for sent in replayed:
        assert sent["body"]["messages"] == original["messages"]
        assert sent["body"]["options"] == original["options"]


@pytest.mark.asyncio
async def test_replay_surfaces_the_difference_between_models(stack):
    row = await record_one(stack)
    run = await replay(row, ["model-a"], stack.upstream_url)

    baseline, other = run.results
    assert baseline.output_text != other.output_text
    assert "fake:latest" in baseline.output_text
    assert "model-a" in other.output_text


@pytest.mark.asyncio
async def test_replay_diagnoses_each_model_independently(stack):
    """A model that starves its own answer must be caught on replay too."""
    body = chat_body(options={"num_ctx": 131_072, "num_predict": 32})
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT, json=body, headers={"x-fake-scenario": "reasoning_starved"}
        )
        assert response.status_code == 200
    await stack.settle()
    row = stack.store.query("SELECT * FROM requests ORDER BY ts DESC")[0]

    run = await replay(row, ["model-a"], stack.upstream_url)
    for result in run.results:
        rules = {d.rule_id for d in result.diagnoses if not d.suppressed_by}
        assert "R004" in rules


@pytest.mark.asyncio
async def test_replay_never_writes_to_the_database(stack):
    row = await record_one(stack)
    before_requests = stack.store.query("SELECT * FROM requests")
    before_diagnoses = stack.store.query("SELECT * FROM diagnoses")

    await replay(row, ["model-a", "model-b"], stack.upstream_url)

    after_requests = stack.store.query("SELECT * FROM requests")
    assert len(after_requests) == len(before_requests)
    assert len(stack.store.query("SELECT * FROM diagnoses")) == len(before_diagnoses)
    # The stored record itself is untouched.
    assert dict(after_requests[0]) == dict(before_requests[0])


@pytest.mark.asyncio
async def test_replay_reports_an_unreachable_model_without_crashing(stack):
    row = await record_one(stack)
    run = await replay(row, ["model-a"], "http://127.0.0.1:9")

    baseline, failed = run.results
    assert baseline.error is None, "the baseline needs no network"
    assert failed.error is not None
    assert failed.status_code is None


@pytest.mark.asyncio
async def test_replay_of_a_streamed_request_reconstructs_the_text(stack):
    row = await record_one(stack, stream=True)
    assert row["stream"] == 1

    run = await replay(row, ["model-a"], stack.upstream_url)
    baseline, other = run.results
    assert "fake:latest" in baseline.output_text
    assert "model-a" in other.output_text
    assert other.completion_tokens is not None


@pytest.mark.asyncio
async def test_render_replay_shows_the_diff(stack):
    from io import StringIO

    from rich.console import Console

    from localdoctor.replay import render_replay

    row = await record_one(stack)
    run = await replay(row, ["model-a"], stack.upstream_url)

    output = StringIO()
    render_replay(Console(file=output, force_terminal=False, no_color=True, width=120), run)
    text = output.getvalue()

    assert "fake:latest (recorded)" in text
    assert "model-a" in text
    assert "% same" in text
    assert "-This is a normal answer from fake:latest." in text
    assert "+This is a normal answer from model-a." in text
