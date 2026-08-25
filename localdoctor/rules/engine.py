"""Rule registration, execution and the suppression graph (spec 6).

No numeric priority — causal suppression instead. The root cause is printed;
suppressed diagnoses are still written to SQLite in full with `suppressed_by`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localdoctor.models import Diagnosis, RequestRecord
from localdoctor.rules.base import Rule
from localdoctor.rules.r001_context_limit import R001
from localdoctor.rules.r002_underuse import R002
from localdoctor.rules.r003_empty import R003
from localdoctor.rules.r004_reasoning_budget import R004

# suppressor -> suppressed
SUPPRESSION: dict[str, tuple[str, ...]] = {
    "R001": ("R003", "R004"),
    "R004": ("R003",),
}

DEFAULT_RULES: tuple[Rule, ...] = (R001(), R002(), R003(), R004())


@dataclass(slots=True)
class EngineResult:
    """`all` goes to SQLite, `root` is the candidate for the terminal."""

    all: list[Diagnosis] = field(default_factory=list)
    root: list[Diagnosis] = field(default_factory=list)
    suppressed: list[Diagnosis] = field(default_factory=list)


class Engine:
    def __init__(self, rules: tuple[Rule, ...] = DEFAULT_RULES) -> None:
        self.rules = rules

    def run(self, record: RequestRecord) -> EngineResult:
        fired: list[Diagnosis] = []
        for rule in self.rules:
            try:
                diagnosis = rule.check(record)
            except Exception:
                # A crashing rule must not take down the proxy or the other rules.
                continue
            if diagnosis is not None:
                fired.append(diagnosis)

        self._apply_suppression(fired)
        result = EngineResult(all=fired)
        for diagnosis in fired:
            (result.suppressed if diagnosis.suppressed_by else result.root).append(diagnosis)
        return result

    @staticmethod
    def _apply_suppression(fired: list[Diagnosis]) -> None:
        by_id = {d.rule_id: d for d in fired}
        changed = True
        while changed:
            changed = False
            for diagnosis in fired:
                # A suppressed rule cannot suppress others: the root cause is upstream.
                if diagnosis.suppressed_by:
                    continue
                for target_id in SUPPRESSION.get(diagnosis.rule_id, ()):
                    target = by_id.get(target_id)
                    if target is None or target is diagnosis or target.suppressed_by:
                        continue
                    target.suppressed_by = diagnosis.rule_id
                    changed = True
