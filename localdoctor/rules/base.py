"""Rule protocol and shared helpers."""

from __future__ import annotations

from typing import Protocol

from localdoctor.models import Confidence, Diagnosis, RequestRecord, Severity


class Rule(Protocol):
    id: str
    title: str
    severity: Severity

    def check(self, record: RequestRecord) -> Diagnosis | None: ...


def diagnose(
    rule: Rule,
    record: RequestRecord,
    confidence: Confidence,
    evidence: dict,
    fix: str,
) -> Diagnosis:
    return Diagnosis(
        rule_id=rule.id,
        request_id=record.id,
        confidence=confidence,
        severity=rule.severity,
        title=rule.title,
        evidence=evidence,
        fix=fix,
        model=record.model,
    )


def fmt_int(value) -> str:
    """4095 -> '4,095'. Used for readable figures, never for values to copy."""
    if not isinstance(value, int):
        return "—"
    return f"{value:,}"
