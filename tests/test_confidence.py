"""The confidence model is the spine of the product (spec 5, spec 11 test 6).

Nothing here may claim more than the signals support.
"""

from __future__ import annotations

import httpx
import pytest

from localdoctor.models import CONFIDENCE_ORDER, ModelFacts, Usage
from localdoctor.report import Reporter
from localdoctor.rules.engine import Engine
from localdoctor.rules.r001_context_limit import R001
from tests.conftest import CHAT, chat_body, make_record


def _max_confidence(diagnoses) -> str | None:
    if not diagnoses:
        return None
    return max(diagnoses, key=lambda d: CONFIDENCE_ORDER.index(d.confidence)).confidence


def test_unknown_window_never_reaches_high():
    """Spec 11, test 6."""
    engine = Engine()
    for prompt in ("short", "x" * 200_000):
        for prompt_tokens in (10, 4096, 8192, 131_072):
            record = make_record(
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=9),
                num_ctx=None,
                num_ctx_source="unknown",
                prompt_text=prompt,
            )
            diagnoses = [d for d in engine.run(record).all if d.rule_id in ("R001", "R002")]
            highest = _max_confidence(diagnoses)
            if highest is not None:
                assert CONFIDENCE_ORDER.index(highest) < CONFIDENCE_ORDER.index("high"), (
                    f"{highest} claimed with an unknown window"
                )


def test_observed_window_downgrades_one_step():
    """An estimated window may never produce more than `medium`."""
    strong = make_record(
        usage=Usage(prompt_tokens=8192, completion_tokens=9),
        num_ctx=8192,
        num_ctx_source="request",
        prompt_text="x" * 200_000,
    )
    assert R001().check(strong).confidence == "high"

    estimated = make_record(
        usage=Usage(prompt_tokens=8192, completion_tokens=9),
        num_ctx=8192,
        num_ctx_source="observed",
        prompt_text="x" * 200_000,
    )
    assert R001().check(estimated).confidence == "medium"


def test_observed_source_is_named_in_the_evidence():
    record = make_record(
        usage=Usage(prompt_tokens=8192, completion_tokens=9),
        num_ctx=8192,
        num_ctx_source="observed",
        prompt_text="x" * 200_000,
    )
    from localdoctor.rules.r001_context_limit import render_lines

    lines = dict(render_lines(R001().check(record).evidence))
    assert "observed, unverified" in lines["Window limit"]


def test_low_confidence_is_never_printed(capsys):
    """Spec 5: low goes to SQLite only."""
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    reporter = Reporter(console=Console(file=output, force_terminal=False, no_color=True))
    record = make_record(
        usage=Usage(prompt_tokens=10, completion_tokens=9),
        num_ctx=32768,
        prompt_text="x" * 6000,
    )
    result = Engine().run(record)
    low = [d for d in result.all if d.confidence == "low"]
    assert low, "expected this shape to produce a low-confidence diagnosis"

    printed = reporter.emit(record, result)
    assert all(d.confidence != "low" for d in printed)
    assert "CONTEXT LIMIT EXCEEDED" not in output.getvalue()


@pytest.mark.asyncio
async def test_unknown_window_end_to_end_stays_below_high(stack):
    """No explicit num_ctx and no model default: the window is genuinely unknown."""
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(3):
            response = await client.post(
                stack.base_url + CHAT,
                json=chat_body(messages=[{"role": "user", "content": "x" * 40_000}]),
                headers={"x-fake-scenario": "truncated"},
            )
            assert response.status_code == 200
            await stack.settle()

    rows = stack.store.query("SELECT rule_id, confidence FROM diagnoses WHERE rule_id = 'R001'")
    for row in rows:
        assert CONFIDENCE_ORDER.index(row["confidence"]) < CONFIDENCE_ORDER.index("high")

    sources = {row["num_ctx_source"] for row in stack.store.query("SELECT num_ctx_source FROM requests")}
    assert sources <= {"unknown", "observed"}


@pytest.mark.asyncio
async def test_a_request_is_never_its_own_observed_ceiling(stack):
    """The observed ceiling must come from earlier traffic, never from the
    request being judged — otherwise S1 would fire on every single call."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT,
            json=chat_body(messages=[{"role": "user", "content": "hello"}]),
        )
        assert response.status_code == 200
    await stack.settle()

    rows = stack.store.query("SELECT num_ctx_source FROM requests")
    assert [row["num_ctx_source"] for row in rows] == ["unknown"]
    assert stack.store.query("SELECT * FROM diagnoses WHERE rule_id = 'R001'") == []
