"""R002 — CONTEXT UNDERUSE · severity: suggestion

Some users deliberately pick a small window for speed, RAM or latency. This is
not an error, it is information. Printed once per model.
"""

from __future__ import annotations

from localdoctor.models import Diagnosis, RequestRecord
from localdoctor.rules.base import diagnose, fmt_int

THRESHOLD_RATIO = 0.25


class R002:
    id = "R002"
    title = "CONTEXT UNDERUSE"
    severity = "suggestion"

    def check(self, record: RequestRecord) -> Diagnosis | None:
        facts = record.model_facts
        if facts is None or not facts.context_length or not record.num_ctx:
            return None
        # Never say "you are underusing it" based on a guessed window.
        if record.num_ctx_source not in ("request", "model_default"):
            return None
        if record.num_ctx >= facts.context_length * THRESHOLD_RATIO:
            return None

        evidence = {
            "effective_num_ctx": record.num_ctx,
            "num_ctx_source": record.num_ctx_source,
            "model_context_length": facts.context_length,
            "ratio": round(record.num_ctx / facts.context_length, 4),
        }
        suggested = min(facts.context_length, max(record.num_ctx * 4, 8192))
        fix = (
            f"num_ctx={suggested} is a safe starting point for this model. "
            "A larger window costs VRAM; ignore this if the small window is deliberate."
        )
        return diagnose(self, record, "certain", evidence, fix)


def render_lines(evidence: dict) -> list[tuple[str, str]]:
    from localdoctor.rules.r001_context_limit import SOURCE_LABEL

    source = SOURCE_LABEL.get(evidence.get("num_ctx_source", "unknown"), "unknown")
    return [
        ("Window in use", f"{fmt_int(evidence.get('effective_num_ctx'))}  (source: {source})"),
        ("Model supports", fmt_int(evidence.get("model_context_length"))),
    ]
