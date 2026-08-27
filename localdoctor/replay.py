"""Replay recorded traffic against other models (phase 3).

"If I move from qwen3.5 to gemma4, does my agent break?" has no answer today.
Because every request is stored with its raw body and headers, that question
becomes a command.

Replay never writes to the database and never touches the stored record. It
sends, measures, diagnoses and diffs — nothing else.
"""

from __future__ import annotations

import difflib
import json
import sqlite3
import time
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.table import Table
from rich.text import Text

from localdoctor.collector import Capture, build_record, enrich
from localdoctor.models import Diagnosis, RequestRecord, new_id, now_iso
from localdoctor.modelinfo import ModelInfo
from localdoctor.report import CONFIDENCE_LABEL
from localdoctor.rules.base import fmt_int
from localdoctor.rules.engine import Engine

# Headers that belong to a single hop and must not be replayed.
SKIP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailer",
    "accept-encoding",
}


@dataclass(slots=True)
class ReplayResult:
    model: str
    baseline: bool = False
    status_code: int | None = None
    output_text: str = ""
    thinking_text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    elapsed_ms: int | None = None
    diagnoses: list[Diagnosis] = field(default_factory=list)
    error: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model} (recorded)" if self.baseline else self.model


@dataclass(slots=True)
class ReplayRun:
    source: sqlite3.Row
    results: list[ReplayResult] = field(default_factory=list)

    @property
    def baseline(self) -> ReplayResult | None:
        return next((r for r in self.results if r.baseline), None)


def _body_for(row: sqlite3.Row, model: str) -> bytes:
    """The stored body with only the model swapped."""
    body = json.loads(row["request_body"])
    body["model"] = model
    return json.dumps(body).encode()


def _headers_for(row: sqlite3.Row) -> dict[str, str]:
    try:
        stored = json.loads(row["request_headers"])
    except Exception:
        stored = {}
    headers = {
        key: value
        for key, value in stored.items()
        if key.lower() not in SKIP_HEADERS
    }
    headers.setdefault("content-type", "application/json")
    return headers


async def _baseline(row: sqlite3.Row, engine: Engine, model_info: ModelInfo) -> ReplayResult:
    """Rebuild what was recorded. No network call is made for this one."""
    capture = Capture(
        id=row["id"],
        ts=row["ts"],
        endpoint=row["endpoint"],
        method="POST",
        request_body=row["request_body"],
        request_headers={},
    )
    capture.status_code = row["status_code"]
    capture.stream = bool(row["stream"])
    capture.buffer = bytearray(row["response_body"] or b"")

    record = build_record(capture)
    await enrich(record, model_info)

    return ReplayResult(
        model=row["model"] or "—",
        baseline=True,
        status_code=row["status_code"],
        output_text=record.output_text,
        thinking_text=record.thinking_text,
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        finish_reason=row["finish_reason"],
        elapsed_ms=row["total_ms"],
        diagnoses=engine.run(record).all,
    )


async def _send(
    client: httpx.AsyncClient,
    upstream: str,
    row: sqlite3.Row,
    model: str,
    engine: Engine,
    model_info: ModelInfo,
) -> ReplayResult:
    url = upstream.rstrip("/") + row["endpoint"]
    body = _body_for(row, model)
    started = time.monotonic()
    try:
        response = await client.post(url, content=body, headers=_headers_for(row))
    except httpx.HTTPError as exc:
        return ReplayResult(model=model, error=f"{type(exc).__name__}: {exc}")
    elapsed = int((time.monotonic() - started) * 1000)

    capture = Capture(
        id=new_id(),
        ts=now_iso(),
        endpoint=row["endpoint"],
        method="POST",
        request_body=body,
        request_headers=_headers_for(row),
    )
    capture.status_code = response.status_code
    capture.response_headers = dict(response.headers)
    content_type = response.headers.get("content-type", "").lower()
    capture.stream = "event-stream" in content_type or "x-ndjson" in content_type
    capture.buffer = bytearray(response.content)
    capture.total_ms = elapsed

    record: RequestRecord = build_record(capture)
    await enrich(record, model_info)

    diagnoses = engine.run(record).all if response.status_code == 200 else []
    return ReplayResult(
        model=model,
        status_code=response.status_code,
        output_text=record.output_text,
        thinking_text=record.thinking_text,
        prompt_tokens=record.usage.prompt_tokens,
        completion_tokens=record.usage.completion_tokens,
        finish_reason=record.usage.finish_reason,
        elapsed_ms=elapsed,
        diagnoses=diagnoses,
        error=None if response.status_code == 200 else f"HTTP {response.status_code}",
    )


async def replay(row: sqlite3.Row, models: list[str], upstream: str) -> ReplayRun:
    engine = Engine()
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        model_info = ModelInfo(upstream, client)
        run = ReplayRun(source=row)
        run.results.append(await _baseline(row, engine, model_info))
        for model in models:
            run.results.append(await _send(client, upstream, row, model, engine, model_info))
    return run


# --- rendering ----------------------------------------------------------


def render_replay(console: Console, run: ReplayRun, show_diff: bool = True) -> None:
    row = run.source
    console.print()
    header = Text()
    header.append("replay  ", style="dim")
    header.append(row["id"][-8:], style="cyan")
    header.append(f"   {row['endpoint']}", style="dim")
    header.append(f"   recorded on {row['model'] or '—'}", style="dim")
    console.print(header)
    console.print()

    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("model")
    table.add_column("in → out", justify="right")
    table.add_column("finish", style="dim")
    table.add_column("ms", justify="right", style="dim")
    table.add_column("output", justify="right", style="dim")
    table.add_column("vs recorded", justify="right")
    table.add_column("findings")

    baseline = run.baseline
    for result in run.results:
        if result.error and result.status_code is None:
            table.add_row(result.label, "—", "—", "—", "—", "—", Text(result.error, style="red"))
            continue
        table.add_row(
            result.label,
            f"{fmt_int(result.prompt_tokens)} → {fmt_int(result.completion_tokens)}",
            result.finish_reason or "—",
            fmt_int(result.elapsed_ms),
            f"{fmt_int(len(result.output_text))} ch",
            _similarity(baseline, result),
            _findings(result),
        )
    console.print(table)

    if show_diff and baseline is not None:
        for result in run.results:
            if result.baseline or result.error:
                continue
            _render_diff(console, baseline, result)
    console.print()


def _similarity(baseline: ReplayResult | None, result: ReplayResult) -> Text:
    if baseline is None or result.baseline:
        return Text("—", style="dim")
    ratio = difflib.SequenceMatcher(None, baseline.output_text, result.output_text).ratio()
    style = "green" if ratio > 0.9 else "yellow" if ratio > 0.5 else "red"
    return Text(f"{ratio * 100:.0f}% same", style=style)


def _findings(result: ReplayResult) -> Text:
    visible = [d for d in result.diagnoses if not d.suppressed_by]
    if not visible:
        return Text("—", style="dim")
    text = Text()
    for index, diagnosis in enumerate(visible):
        if index:
            text.append("  ")
        label = CONFIDENCE_LABEL.get(diagnosis.confidence, diagnosis.confidence)
        style = "dim" if diagnosis.confidence == "low" else "red"
        text.append(f"{diagnosis.rule_id} {label}", style=style)
    return text


def _render_diff(console: Console, baseline: ReplayResult, result: ReplayResult) -> None:
    diff = list(
        difflib.unified_diff(
            baseline.output_text.splitlines(),
            result.output_text.splitlines(),
            fromfile=f"recorded {baseline.model}",
            tofile=result.model,
            lineterm="",
            n=1,
        )
    )
    console.print()
    if not diff:
        console.print(Text(f"  {result.model}: output identical to the recording", style="dim"))
        return
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            console.print(Text("  " + line, style="bold dim"))
        elif line.startswith("@@"):
            console.print(Text("  " + line, style="cyan"))
        elif line.startswith("+"):
            console.print(Text("  " + line, style="green"))
        elif line.startswith("-"):
            console.print(Text("  " + line, style="red"))
        else:
            console.print(Text("  " + line, style="dim"))
