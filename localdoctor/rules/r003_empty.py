"""R003 — EMPTY OUTPUT · certain

If tokens were generated and yet the content handed back is empty, there is an
undeniable mismatch. This one really is certain.
"""

from __future__ import annotations

from localdoctor.models import Diagnosis, RequestRecord
from localdoctor.rules.base import diagnose, fmt_int

# Embedding endpoints never carry text content; R003 does not apply there.
SKIP_ENDPOINTS = {"/api/embed"}


class R003:
    id = "R003"
    title = "EMPTY OUTPUT"
    severity = "error"

    def check(self, record: RequestRecord) -> Diagnosis | None:
        if record.endpoint in SKIP_ENDPOINTS:
            return None
        if record.status_code != 200:
            return None
        completion_tokens = record.usage.completion_tokens
        if not completion_tokens or completion_tokens <= 0:
            return None
        if record.output_text.strip():
            return None

        evidence = {
            "status_code": record.status_code,
            "completion_tokens": completion_tokens,
            "output_chars": len(record.output_text),
            "finish_reason": record.usage.finish_reason,
        }
        fix = (
            "The chat template or stop-token configuration may not match what the "
            "model generated. Inspect the raw response with `localdoctor log`."
        )
        return diagnose(self, record, "certain", evidence, fix)


def render_lines(evidence: dict) -> list[tuple[str, str]]:
    return [
        ("Tokens generated", fmt_int(evidence.get("completion_tokens"))),
        ("Content returned", f"{fmt_int(evidence.get('output_chars'))} chars (empty)"),
        ("Finish reason", str(evidence.get("finish_reason") or "—")),
    ]
