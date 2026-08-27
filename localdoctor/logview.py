"""Rendering for `localdoctor log` and `localdoctor show` (phase 2).

The terminal during `serve` only shows what crosses the confidence threshold.
These two commands are where everything else lives — including the `low`
confidence guesses that are recorded but never printed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text

from localdoctor.report import CONFIDENCE_LABEL, RENDERERS, SEVERITY_STYLE
from localdoctor.rules.base import fmt_int

CONFIDENCE_STYLE = {
    "certain": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "dim",
}

PREVIEW_CHARS = 600


def short_id(request_id: str) -> str:
    """The tail of a ULID is its random half — the part that actually differs."""
    return request_id[-8:]


def local_time(ts: str, with_date: bool = False) -> str:
    try:
        moment = datetime.fromisoformat(ts).astimezone()
    except ValueError:
        return ts
    return moment.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def render_log(console: Console, rows: list[sqlite3.Row], store) -> None:
    if not rows:
        console.print(Text("no matching requests", style="dim"))
        return

    table = Table(box=None, pad_edge=False, header_style="dim", expand=False)
    table.add_column("id", style="cyan")
    table.add_column("time", style="dim")
    table.add_column("model")
    table.add_column("endpoint", style="dim")
    table.add_column("tokens", justify="right")
    table.add_column("ms", justify="right", style="dim")
    table.add_column("findings")

    for row in rows:
        diagnoses = store.diagnoses_for(row["id"])
        tokens = f"{fmt_int(row['prompt_tokens'])} → {fmt_int(row['completion_tokens'])}"
        table.add_row(
            short_id(row["id"]),
            local_time(row["ts"]),
            row["model"] or "—",
            row["endpoint"],
            tokens,
            fmt_int(row["total_ms"]),
            _findings_summary(diagnoses),
        )
    console.print()
    console.print(table)
    console.print()
    console.print(
        Text(f"  {len(rows)} shown · localdoctor show <id> for detail", style="dim")
    )


def _findings_summary(diagnoses: list[sqlite3.Row]) -> Text:
    if not diagnoses:
        return Text("—", style="dim")
    text = Text()
    for index, diagnosis in enumerate(diagnoses):
        if index:
            text.append("  ")
        if diagnosis["suppressed_by"]:
            text.append(
                f"{diagnosis['rule_id']}↳{diagnosis['suppressed_by']}", style="dim"
            )
            continue
        style = CONFIDENCE_STYLE.get(diagnosis["confidence"], "")
        label = CONFIDENCE_LABEL.get(diagnosis["confidence"], diagnosis["confidence"])
        text.append(f"{diagnosis['rule_id']} {label}", style=style)
    return text


def render_show(console: Console, row: sqlite3.Row, diagnoses: list[sqlite3.Row], full: bool) -> None:
    console.print()
    header = Text()
    header.append("request  ", style="dim")
    header.append(row["id"], style="cyan")
    console.print(header)

    stream = "yes" if row["stream"] else "no"
    timings = f"ttft {fmt_int(row['ttft_ms'])}ms · total {fmt_int(row['total_ms'])}ms"
    if row["chunk_count"]:
        timings += f" · {fmt_int(row['chunk_count'])} chunks"

    for label, value in [
        ("time", local_time(row["ts"], with_date=True)),
        ("endpoint", f"{row['endpoint']}   stream: {stream}"),
        ("model", row["model"] or "—"),
        ("status", f"{row['status_code']}   {timings}"),
        (
            "tokens",
            f"prompt {fmt_int(row['prompt_tokens'])} → completion "
            f"{fmt_int(row['completion_tokens'])}   finish: {row['finish_reason'] or '—'}",
        ),
        ("window", f"{fmt_int(row['num_ctx'])}  (source: {row['num_ctx_source']})"),
    ]:
        line = Text("  ")
        line.append(f"{label:<10}", style="dim")
        line.append(str(value))
        console.print(line)

    console.print()
    if not diagnoses:
        console.print(Text("  no diagnoses — this request looked healthy", style="dim"))
    for diagnosis in diagnoses:
        _render_diagnosis(console, diagnosis)

    _render_body(console, "request body", row["request_body"], full)
    _render_body(console, "response body", row["response_body"], full)
    console.print()


def _render_diagnosis(console: Console, diagnosis: sqlite3.Row) -> None:
    icon, style = SEVERITY_STYLE.get(diagnosis["severity"], ("·", "dim"))
    suppressed = diagnosis["suppressed_by"]
    if suppressed:
        icon, style = "·", "dim"

    header = Text("  ")
    header.append(f"{icon} {diagnosis['rule_id']}", style=style)
    header.append(
        f"   confidence: {CONFIDENCE_LABEL.get(diagnosis['confidence'], diagnosis['confidence'])}",
        style=CONFIDENCE_STYLE.get(diagnosis["confidence"], "dim"),
    )
    if suppressed:
        header.append(f"   suppressed by {suppressed}", style="dim")
    if diagnosis["confidence"] == "low":
        header.append("   (recorded only, never printed live)", style="dim italic")
    console.print(header)

    try:
        evidence = json.loads(diagnosis["evidence"])
    except Exception:
        evidence = {}
    renderer = RENDERERS.get(diagnosis["rule_id"])
    if renderer and evidence:
        for label, value in renderer(evidence):
            line = Text("     ")
            line.append(f"{label:<22}", style="dim")
            line.append(value)
            console.print(line)
    console.print(Text(f"     ► {diagnosis['fix']}", style="dim" if suppressed else "green"))
    console.print()


def _render_body(console: Console, label: str, raw: bytes | None, full: bool) -> None:
    if not raw:
        return
    text = raw.decode("utf-8", errors="replace")
    console.print(Text(f"  {label}   {fmt_int(len(raw))} bytes", style="dim"))
    if not full and len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + f"\n     … {fmt_int(len(raw) - PREVIEW_CHARS)} more bytes (--full)"
    for line in text.splitlines():
        console.print(Text("     " + line, style="dim"))
    console.print()
