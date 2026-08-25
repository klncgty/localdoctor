"""R001 — CONTEXT LIMIT EXCEEDED

Confidence can NEVER be `certain`. A prompt_eval_count pinned to the window
boundary has two explanations: the user really did send exactly that many
tokens, or the excess was cut. On its own the signal cannot tell them apart.

Truncation does not always pin to the wall. Ollama runs llama-server with
`--context-shift`, which drops part of an oversized prompt instead of filling
the window, so the reported count can land far below num_ctx. That shape is
caught by S2 and S3 together rather than by S1.

Forbidden claims: "your system prompt is gone", "the first 78% of your prompt
was dropped", "your instruction never reached the model". Which part was cut is
not visible from here.
"""

from __future__ import annotations

from localdoctor.models import Diagnosis, RequestRecord, downgrade
from localdoctor.rules.base import diagnose, fmt_int

WALL_TOLERANCE = 8

SOURCE_LABEL = {
    "request": "set in request",
    "model_default": "model default",
    "observed": "observed, unverified",
    "unknown": "unknown",
}


class R001:
    id = "R001"
    title = "CONTEXT LIMIT EXCEEDED"
    severity = "error"

    def check(self, record: RequestRecord) -> Diagnosis | None:
        prompt_tokens = record.usage.prompt_tokens
        if prompt_tokens is None:
            return None

        num_ctx = record.num_ctx
        min_sent = record.min_prompt_tokens

        # S1: pinned to the window boundary
        s1 = num_ctx is not None and prompt_tokens >= num_ctx - WALL_TOLERANCE
        # S2: the guaranteed lower bound of what we sent already exceeds the window
        s2 = num_ctx is not None and min_sent > num_ctx
        # S3: the model read less than the guaranteed lower bound of what we sent
        s3 = min_sent > 0 and prompt_tokens < min_sent

        if not (s1 or s2 or s3):
            return None

        if s1 and (s2 or s3):
            confidence = "high"
        elif s2 and s3:
            # Two measurements taken against different operands agree: we proved
            # the input cannot fit the window, AND we proved the model read less
            # than we sent. This is the shape truncation takes when the server
            # runs with context shift, where the count never pins to the wall.
            confidence = "high"
        elif s1:
            confidence = "medium"
        else:
            # S2 or S3 alone is heuristic only.
            confidence = "low"

        if record.num_ctx_source == "observed":
            confidence = downgrade(confidence)

        evidence = {
            "prompt_eval_count": prompt_tokens,
            "effective_num_ctx": num_ctx,
            "num_ctx_source": record.num_ctx_source,
            "min_sent_tokens": min_sent,
            "model_context_length": record.model_facts.context_length if record.model_facts else None,
            "signals": [name for name, hit in (("S1", s1), ("S2", s2), ("S3", s3)) if hit],
        }
        return diagnose(self, record, confidence, evidence, _fix(record))


def _fix(record: RequestRecord) -> str:
    suggestion = suggest_num_ctx(record)
    ceiling = record.model_facts.context_length if record.model_facts else None
    if suggestion and ceiling:
        return f"Try num_ctx={suggestion} (this model supports {fmt_int(ceiling)}), or split the input."
    if suggestion:
        return f"Try num_ctx={suggestion}, or split the input."
    return "Raise num_ctx toward the model's context_length, or split the input."


def suggest_num_ctx(record: RequestRecord) -> int | None:
    """Smallest power of two above both the current window and the sent lower bound."""
    ceiling = record.model_facts.context_length if record.model_facts else None
    current = record.num_ctx or 0
    target = max(current * 2, record.min_prompt_tokens * 2, 8192)
    value = 1
    while value < target:
        value *= 2
    if ceiling:
        value = min(value, ceiling)
        if value <= current:
            return None
    return value


def render_lines(evidence: dict) -> list[tuple[str, str]]:
    """Measured values only."""
    source = SOURCE_LABEL.get(evidence.get("num_ctx_source", "unknown"), "unknown")
    return [
        ("Model read", f"{fmt_int(evidence.get('prompt_eval_count'))} tokens"),
        ("Window limit", f"{fmt_int(evidence.get('effective_num_ctx'))}  (source: {source})"),
        ("Sent (lower bound)", f">= {fmt_int(evidence.get('min_sent_tokens'))} tokens"),
    ]


# The strongest claim this rule is allowed to make.
HINT = "Input appears to exceed the context limit; what was cut is not visible from here."
