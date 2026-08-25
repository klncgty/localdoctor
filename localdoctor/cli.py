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
from localdoctor.models import ModelFacts, RequestRecord, new_id, now_iso
from localdoctor.modelinfo import ModelInfo
from localdoctor.proxy import Settings, create_app
from localdoctor.rules.base import fmt_int
from localdoctor.rules.r002_underuse import R002
from localdoctor.store import DEFAULT_DB

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
