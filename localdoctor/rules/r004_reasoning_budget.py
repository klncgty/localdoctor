"""R004 — REASONING BUDGET STARVATION · high

The reasoning block ate the generation budget and nothing was left for the
final answer.
"""

from __future__ import annotations

from localdoctor.estimate import min_tokens
from localdoctor.models import Diagnosis, RequestRecord
from localdoctor.normalize import requested_num_predict
from localdoctor.rules.base import diagnose, fmt_int
from localdoctor.rules.reasoning import detect_reasoning, strip_reasoning

BUDGET_RATIO = 0.8
MIN_NUM_PREDICT_FIX = 4096


class R004:
    id = "R004"
    title = "REASONING BUDGET STARVATION"
    severity = "error"

    def check(self, record: RequestRecord) -> Diagnosis | None:
        span = detect_reasoning(record.output_text, record.model, record.thinking_text)
        if span is None:
            # Models with hidden reasoning land here. Skipped silently.
            return None

        final_answer = strip_reasoning(record.output_text)
        if final_answer.strip():
            return None

        num_predict = requested_num_predict(record.request_json)
        span_min_tokens = min_tokens(span.text)
        hit_length = record.usage.finish_reason == "length"
        # Lower bound again: if even the guaranteed minimum covers 80% of the
        # budget, the span certainly does.
        eats_budget = bool(num_predict) and span_min_tokens > num_predict * BUDGET_RATIO

        if not (hit_length or eats_budget):
            return None

        evidence = {
            "detector": span.detector,
            "reasoning_chars": span.char_len,
            "reasoning_min_tokens": span_min_tokens,
            "num_predict": num_predict,
            "finish_reason": record.usage.finish_reason,
            "completion_tokens": record.usage.completion_tokens,
            "final_answer_chars": len(final_answer.strip()),
        }
        suggested = max(MIN_NUM_PREDICT_FIX, (num_predict or 0) * 2)
        fix = f"Set num_predict>={suggested}, or disable thinking with think=false."
        return diagnose(self, record, "high", evidence, fix)


def render_lines(evidence: dict) -> list[tuple[str, str]]:
    lines = [
        (
            "Reasoning block",
            f">= {fmt_int(evidence.get('reasoning_min_tokens'))} tokens ({evidence.get('detector')})",
        ),
        ("Final answer", f"{fmt_int(evidence.get('final_answer_chars'))} chars (empty)"),
        ("Finish reason", str(evidence.get("finish_reason") or "—")),
    ]
    if evidence.get("num_predict"):
        lines.append(("Generation budget", f"num_predict={fmt_int(evidence.get('num_predict'))}"))
    return lines
