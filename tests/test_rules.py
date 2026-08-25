"""Each rule fires on the right signals, and suppression picks the root cause."""

from __future__ import annotations

import json

import httpx
import pytest

from localdoctor.models import ModelFacts, Usage
from localdoctor.rules.engine import Engine
from localdoctor.rules.r001_context_limit import R001
from localdoctor.rules.r002_underuse import R002
from localdoctor.rules.r003_empty import R003
from localdoctor.rules.r004_reasoning_budget import R004
from tests.conftest import CHAT, chat_body, make_record


# --- R001 ---------------------------------------------------------------


def test_r001_s1_alone_is_medium():
    record = make_record(usage=Usage(prompt_tokens=8192, completion_tokens=9), num_ctx=8192)
    diagnosis = R001().check(record)
    assert diagnosis.confidence == "medium"
    assert diagnosis.evidence["signals"] == ["S1"]


def test_r001_s1_plus_s2_is_high():
    record = make_record(
        usage=Usage(prompt_tokens=8192, completion_tokens=9),
        num_ctx=8192,
        prompt_text="x" * 200_000,
    )
    diagnosis = R001().check(record)
    assert diagnosis.confidence == "high"
    assert set(diagnosis.evidence["signals"]) >= {"S1", "S2"}


def test_r001_context_shift_truncation_is_high():
    """Measured against real Ollama 0.32.15 (llama-server --context-shift):
    the same 40,000-char prompt reported 8,017 tokens in a 131,072 window and
    only 258 in a 512 window. Truncation without pinning to the wall."""
    record = make_record(
        usage=Usage(prompt_tokens=258, completion_tokens=16, finish_reason="length"),
        num_ctx=512,
        num_ctx_source="request",
        prompt_text="lorem ipsum dolor sit amet " * 1600,
    )
    diagnosis = R001().check(record)
    assert diagnosis.evidence["signals"] == ["S2", "S3"]
    assert diagnosis.confidence == "high"
    assert diagnosis.evidence["prompt_eval_count"] == 258


def test_r001_s3_alone_is_low_and_never_printed():
    record = make_record(
        usage=Usage(prompt_tokens=10, completion_tokens=9),
        num_ctx=32768,
        prompt_text="x" * 6000,
    )
    diagnosis = R001().check(record)
    assert diagnosis.evidence["signals"] == ["S3"]
    assert diagnosis.confidence == "low"


def test_r001_is_never_certain():
    """The signal has two explanations, so the top level is out of reach."""
    for prompt_tokens in (8184, 8192, 8000):
        for prompt in ("x" * 200_000, "short"):
            record = make_record(
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=9),
                num_ctx=8192,
                prompt_text=prompt,
            )
            diagnosis = R001().check(record)
            if diagnosis:
                assert diagnosis.confidence != "certain"


def test_r001_silent_when_nothing_is_wrong():
    assert R001().check(make_record()) is None


# --- R002 ---------------------------------------------------------------


def test_r002_fires_below_a_quarter_of_the_window():
    record = make_record(num_ctx=4096, num_ctx_source="model_default")
    diagnosis = R002().check(record)
    assert diagnosis.severity == "suggestion"
    assert diagnosis.evidence["model_context_length"] == 32768


def test_r002_silent_on_a_guessed_window():
    """We never say "you are underusing it" based on an estimate."""
    assert R002().check(make_record(num_ctx=4096, num_ctx_source="observed")) is None
    assert R002().check(make_record(num_ctx=None, num_ctx_source="unknown")) is None


# --- R003 ---------------------------------------------------------------


def test_r003_fires_on_tokens_without_content():
    record = make_record(usage=Usage(prompt_tokens=10, completion_tokens=7), output_text="  \n ")
    diagnosis = R003().check(record)
    assert diagnosis.confidence == "certain"
    assert diagnosis.evidence["completion_tokens"] == 7


def test_r003_silent_without_generated_tokens():
    record = make_record(usage=Usage(prompt_tokens=10, completion_tokens=0), output_text="")
    assert R003().check(record) is None


def test_r003_skips_embeddings():
    record = make_record(
        endpoint="/api/embed", usage=Usage(prompt_tokens=4, completion_tokens=4), output_text=""
    )
    assert R003().check(record) is None


# --- R004 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "output_text, thinking_text, detector",
    [
        ("", "thinking hard. " * 100, "native_thinking_field"),
        ("<think>" + "thinking hard. " * 100 + "</think>", "", "qwen_think_tags"),
        ("<reasoning>" + "thinking hard. " * 100 + "</reasoning>", "", "generic_xml_tags"),
    ],
)
def test_r004_detects_each_reasoning_shape(output_text, thinking_text, detector):
    record = make_record(
        usage=Usage(prompt_tokens=10, completion_tokens=128, finish_reason="length"),
        output_text=output_text,
        thinking_text=thinking_text,
    )
    diagnosis = R004().check(record)
    assert diagnosis.confidence == "high"
    assert diagnosis.evidence["detector"] == detector


def test_r004_silent_when_a_final_answer_exists():
    record = make_record(
        usage=Usage(prompt_tokens=10, completion_tokens=128, finish_reason="length"),
        output_text="here is the answer",
        thinking_text="thinking hard. " * 100,
    )
    assert R004().check(record) is None


def test_r004_silent_on_hidden_reasoning():
    """A model that never exposes reasoning must not be reported as a failure."""
    record = make_record(
        usage=Usage(prompt_tokens=10, completion_tokens=128, finish_reason="length"),
        output_text="",
        thinking_text="",
    )
    assert R004().check(record) is None


# --- suppression --------------------------------------------------------


def test_r001_suppresses_r003_and_r004():
    record = make_record(
        usage=Usage(prompt_tokens=8192, completion_tokens=128, finish_reason="length"),
        num_ctx=8192,
        prompt_text="x" * 200_000,
        output_text="",
        thinking_text="thinking hard. " * 100,
    )
    result = Engine().run(record)
    fired = {d.rule_id: d for d in result.all}
    assert {"R001", "R003", "R004"} <= set(fired)
    assert [d.rule_id for d in result.root] == ["R001"]
    assert fired["R003"].suppressed_by == "R001"
    assert fired["R004"].suppressed_by == "R001"


def test_r004_suppresses_r003():
    record = make_record(
        usage=Usage(prompt_tokens=10, completion_tokens=128, finish_reason="length"),
        output_text="",
        thinking_text="thinking hard. " * 100,
        num_ctx=8192,
    )
    result = Engine().run(record)
    fired = {d.rule_id: d for d in result.all}
    assert fired["R003"].suppressed_by == "R004"
    assert fired["R004"].suppressed_by is None


@pytest.mark.asyncio
async def test_context_shift_truncation_is_reported_end_to_end(stack):
    """The product's flagship case must actually reach the terminal."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT,
            json=chat_body(
                messages=[{"role": "user", "content": "lorem ipsum dolor sit amet " * 1600}],
                options={"num_ctx": 512},
            ),
            headers={"x-fake-scenario": "context_shift"},
        )
    assert response.status_code == 200
    await stack.settle()

    printed = stack.printed()
    assert "CONTEXT LIMIT EXCEEDED" in printed
    assert "confidence: high" in printed

    row = stack.store.query("SELECT * FROM diagnoses WHERE rule_id = 'R001'")[0]
    assert row["confidence"] == "high"
    assert json.loads(row["evidence"])["signals"] == ["S2", "S3"]


@pytest.mark.asyncio
async def test_suppressed_diagnosis_is_stored_but_not_printed(stack):
    """Spec 11, test 7, end to end."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT,
            json=chat_body(
                messages=[{"role": "user", "content": "x" * 20_000}],
                options={"num_ctx": 512},
            ),
            headers={"x-fake-scenario": "truncated_and_empty"},
        )
    assert response.status_code == 200
    await stack.settle()

    printed = stack.printed()
    assert "CONTEXT LIMIT EXCEEDED" in printed
    assert "EMPTY OUTPUT" not in printed
    assert "related: R003" in printed

    rows = {row["rule_id"]: row for row in stack.store.query("SELECT * FROM diagnoses")}
    assert rows["R001"]["suppressed_by"] is None
    assert rows["R003"]["suppressed_by"] == "R001"
    assert json.loads(rows["R001"]["evidence"])["prompt_eval_count"] == 512
