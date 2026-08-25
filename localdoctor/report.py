"""rich output: confidence threshold, rate limiting, root cause (spec 8).

Healthy requests print nothing. Silence means everything is fine — but even in
silence the request is written to SQLite.
"""

from __future__ import annotations

import time
from datetime import datetime

from rich.console import Console
from rich.text import Text

from localdoctor.models import CONFIDENCE_ORDER, PRINT_THRESHOLD, Diagnosis, RequestRecord
from localdoctor.rules import r001_context_limit, r002_underuse, r003_empty, r004_reasoning_budget
from localdoctor.rules.engine import EngineResult

CONFIDENCE_LABEL = {
    "certain": "certain",
    "high": "high",
    "medium": "likely",
    "low": "low",
}

SEVERITY_STYLE = {
    "error": ("⚠", "bold red"),
    "warning": ("⚠", "bold yellow"),
    "suggestion": ("·", "dim"),
}

# Each rule renders its own evidence lines; report only dispatches.
RENDERERS = {
    "R001": r001_context_limit.render_lines,
    "R002": r002_underuse.render_lines,
    "R003": r003_empty.render_lines,
    "R004": r004_reasoning_budget.render_lines,
}

HINTS = {"R001": r001_context_limit.HINT}

# Rules printed at most once per model (spec 6, R002).
ONCE_PER_MODEL = {"R002"}

RATE_LIMIT_SECONDS = 60
LABEL_WIDTH = 22


class Reporter:
    """No module-level state: the proxy owns a single instance."""

    def __init__(self, console: Console | None = None, quiet: bool = False) -> None:
        self.console = console or Console()
        self.quiet = quiet
        self._last_printed: dict[tuple[str | None, str], float] = {}

    def _allowed(self, diagnosis: Diagnosis) -> bool:
        key = (diagnosis.model, diagnosis.rule_id)
        now = time.monotonic()
        last = self._last_printed.get(key)
        if last is not None:
            if diagnosis.rule_id in ONCE_PER_MODEL:
                return False
            if now - last < RATE_LIMIT_SECONDS:
                return False
        self._last_printed[key] = now
        return True

    def emit(self, record: RequestRecord, result: EngineResult) -> list[Diagnosis]:
        """Returns what was actually printed, so tests can assert on it."""
        if self.quiet:
            return []
        printed: list[Diagnosis] = []
        for diagnosis in result.root:
            # `low` never reaches the terminal.
            if CONFIDENCE_ORDER.index(diagnosis.confidence) < CONFIDENCE_ORDER.index(PRINT_THRESHOLD):
                continue
            if not self._allowed(diagnosis):
                continue
            self._render(diagnosis, result)
            printed.append(diagnosis)
        return printed

    def _render(self, diagnosis: Diagnosis, result: EngineResult) -> None:
        icon, style = SEVERITY_STYLE.get(diagnosis.severity, ("·", "dim"))
        dim = "dim" if diagnosis.severity == "suggestion" else None
        clock = datetime.now().strftime("%H:%M:%S")
        confidence = CONFIDENCE_LABEL.get(diagnosis.confidence, diagnosis.confidence)

        header = Text()
        header.append(f"{icon}  {diagnosis.title}", style=style)
        header.append(f"   {diagnosis.model or '—'}", style="cyan")
        header.append(f"   {clock}", style="dim")
        header.append(f"   confidence: {confidence}", style="dim")
        self.console.print(header)

        renderer = RENDERERS.get(diagnosis.rule_id)
        if renderer:
            for label, value in renderer(diagnosis.evidence):
                line = Text("   ")
                line.append(f"{label:<{LABEL_WIDTH}}", style="dim")
                line.append(value)
                self.console.print(line, style=dim)

        hint = HINTS.get(diagnosis.rule_id)
        if hint:
            self.console.print(Text(f"   {hint}", style="dim italic"))

        self.console.print(
            Text(f"   ► {diagnosis.fix}", style="green" if dim is None else "dim")
        )

        related = [d.rule_id for d in result.suppressed if d.suppressed_by == diagnosis.rule_id]
        if related:
            self.console.print(Text(f"   related: {', '.join(related)}", style="dim"))
        self.console.print()
