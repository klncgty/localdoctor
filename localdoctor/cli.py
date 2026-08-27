"""Command line surface. Phase 1 ships exactly two commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.text import Text

from localdoctor import __version__
from localdoctor.dashboard import create_dashboard_app
from localdoctor.logview import render_log, render_show
from localdoctor.models import ModelFacts, RequestRecord, new_id, now_iso
from localdoctor.modelinfo import ModelInfo
from localdoctor.proxy import Settings, create_app
from localdoctor.replay import render_replay
from localdoctor.replay import replay as run_replay
from localdoctor.rules.base import fmt_int
from localdoctor.rules.r002_underuse import R002
from localdoctor.store import DEFAULT_DB, AmbiguousId, Store

app = typer.Typer(
    add_completion=False,
    help="A diagnostic proxy for local LLM servers. Point your base URL here.",
)
console = Console()


@app.command()
def serve(
    port: int = typer.Option(11435, help="Port LocalDoctor listens on."),
    upstream: str = typer.Option("http://localhost:11434", help="The real LLM server."),
    host: str = typer.Option("127.0.0.1", help="Interface to bind. Local by default."),
    quiet: bool = typer.Option(False, "--quiet", help="Print nothing; still record everything."),
    record: bool = typer.Option(False, "--record", help="Also record chunk-level timing."),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite database path."),
) -> None:
    """Run the diagnostic proxy."""
    settings = Settings(upstream=upstream, quiet=quiet, record_chunks=record, db_path=db)
    if not quiet:
        banner = Text()
        banner.append(f"localdoctor {__version__}", style="bold")
        banner.append(f"  http://{host}:{port}", style="cyan")
        banner.append("  →  ", style="dim")
        banner.append(upstream, style="cyan")
        console.print(banner)
        console.print(
            Text(f"  recording to {db}   ·   silence means nothing is wrong", style="dim")
        )
        console.print()
    uvicorn.run(create_app(settings, console=console), host=host, port=port, log_level="warning")


def _open_store(db: Path) -> Store:
    """Open an existing database, or explain that there is nothing to read yet."""
    path = Path(db).expanduser()
    if not path.exists():
        console.print(Text(f"no database at {path}", style="yellow"))
        console.print(Text("run `localdoctor serve` and send some traffic first", style="dim"))
        raise typer.Exit(1)
    return Store(path)


@app.command()
def log(
    limit: int = typer.Option(20, "--limit", "-n", help="How many requests to list."),
    model: str = typer.Option(None, help="Only requests whose model matches."),
    rule: str = typer.Option(None, help="Only requests that fired this rule, e.g. R001."),
    endpoint: str = typer.Option(None, help="Only this endpoint."),
    show_all: bool = typer.Option(False, "--all", help="Include healthy requests too."),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite database path."),
) -> None:
    """List recorded requests, including the low-confidence guesses never printed live."""
    store = _open_store(db)
    try:
        rows = store.recent(
            limit=limit, model=model, rule=rule, endpoint=endpoint, only_findings=not show_all
        )
        render_log(console, rows, store)
    finally:
        store.close()


@app.command()
def show(
    ident: str = typer.Argument(..., help="Request id, or a unique part of one."),
    full: bool = typer.Option(False, "--full", help="Print bodies in full, not a preview."),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite database path."),
) -> None:
    """Everything recorded about a single request."""
    store = _open_store(db)
    try:
        row = _lookup(store, ident)
        render_show(console, row, store.diagnoses_for(row["id"]), full)
    finally:
        store.close()


@app.command()
def dashboard(
    port: int = typer.Option(11436, help="Port the dashboard listens on."),
    host: str = typer.Option("127.0.0.1", help="Interface to bind. Local by default."),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite database path."),
) -> None:
    """Serve the embedded dashboard. Nothing is loaded from outside this machine."""
    store = _open_store(db)
    console.print(Text(f"localdoctor dashboard  http://{host}:{port}", style="cyan"))
    console.print(Text(f"  reading {Path(db).expanduser()}", style="dim"))
    console.print()
    try:
        uvicorn.run(create_dashboard_app(store), host=host, port=port, log_level="warning")
    finally:
        store.close()


@app.command()
def replay(
    ident: str = typer.Argument(..., help="Request id, or a unique part of one."),
    model: list[str] = typer.Option(
        None, "--model", "-m", help="Model to replay against. Repeat for several."
    ),
    upstream: str = typer.Option("http://localhost:11434", help="Server to replay against."),
    diff: bool = typer.Option(True, "--diff/--no-diff", help="Show the output diff."),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite database path."),
) -> None:
    """Send a recorded request to other models and diff what comes back.

    The stored record is never modified and replay results are not written to
    the database. Only the model field of the request is changed.
    """
    store = _open_store(db)
    try:
        row = _lookup(store, ident)
        if not row["request_body"]:
            console.print(Text("this request has no stored body to replay", style="yellow"))
            raise typer.Exit(1)
        models = list(model or [])
        if not models:
            console.print(
                Text("no --model given; replaying against the recorded model", style="dim")
            )
            if row["model"]:
                models = [row["model"]]
        run = asyncio.run(run_replay(row, models, upstream))
        render_replay(console, run, show_diff=diff)
    finally:
        store.close()


def _lookup(store: Store, ident: str):
    try:
        row = store.find_request(ident)
    except AmbiguousId as exc:
        console.print(Text(f"`{ident}` matches {len(exc.matches)} requests:", style="yellow"))
        for match in exc.matches[:10]:
            console.print(Text(f"  {match}", style="dim"))
        raise typer.Exit(1) from None
    if row is None:
        console.print(Text(f"no request matching `{ident}`", style="yellow"))
        raise typer.Exit(1)
    return row


@app.command()
def doctor(
    upstream: str = typer.Option("http://localhost:11434", help="The real LLM server."),
) -> None:
    """One-shot check without waiting for traffic.

    Is the server up, which models are installed, what context_length do they
    have, what is their default num_ctx. Applies R002 to every model.
    """
    raise SystemExit(asyncio.run(_doctor(upstream.rstrip("/"))))


async def _doctor(upstream: str) -> int:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            version = await client.get(f"{upstream}/api/version")
            tags = await client.get(f"{upstream}/api/tags")
        except httpx.HTTPError as exc:
            console.print(Text(f"⚠  {upstream} unreachable ({type(exc).__name__})", style="bold red"))
            return 1

        server_version = _json(version).get("version", "unknown")
        console.print(Text(f"server   {upstream}  ·  version {server_version}", style="dim"))

        models = [m.get("name") for m in _json(tags).get("models", []) if m.get("name")]
        if not models:
            console.print(Text("no models installed", style="yellow"))
            return 0

        info = ModelInfo(upstream, client)
        table = Table(box=None, pad_edge=False, header_style="dim")
        table.add_column("model")
        table.add_column("context_length", justify="right")
        table.add_column("default num_ctx", justify="right")
        table.add_column("capabilities", style="dim")

        findings = []
        for name in models:
            facts = await info.get(name)
            facts = facts or ModelFacts(name=name)
            default = (
                fmt_int(facts.default_num_ctx)
                if facts.default_num_ctx
                else Text("unset (server default)", style="dim")
            )
            table.add_row(
                name,
                fmt_int(facts.context_length),
                default if isinstance(default, str) else default.plain,
                ", ".join(facts.capabilities) or "—",
            )
            finding = _check_underuse(name, facts)
            if finding:
                findings.append(finding)

        console.print()
        console.print(table)
        console.print()

        for text in findings:
            console.print(text)
        if not findings:
            console.print(Text("no context findings", style="dim"))
        console.print(
            Text(
                "\nA model with no default num_ctx uses the server-wide setting "
                "(OLLAMA_CONTEXT_LENGTH), which is not visible from here.",
                style="dim italic",
            )
        )
        return 0


def _check_underuse(name: str, facts: ModelFacts) -> Text | None:
    """Run R002 against a synthetic record built from the model's own defaults."""
    if not facts.default_num_ctx or not facts.context_length:
        return None
    record = RequestRecord(
        id=new_id(),
        ts=now_iso(),
        endpoint="/api/chat",
        request_body=b"{}",
        request_headers={},
        stream=False,
        model=name,
        status_code=200,
        num_ctx=facts.default_num_ctx,
        num_ctx_source="model_default",
        model_facts=facts,
    )
    diagnosis = R002().check(record)
    if diagnosis is None:
        return None
    text = Text()
    text.append("·  CONTEXT UNDERUSE", style="dim")
    text.append(f"  {name}", style="cyan")
    text.append(
        f"\n   window {fmt_int(facts.default_num_ctx)} of {fmt_int(facts.context_length)}"
        f"\n   ► {diagnosis.fix}",
        style="dim",
    )
    return text


def _json(response: httpx.Response) -> dict:
    try:
        value = response.json()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    app()
