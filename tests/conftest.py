"""Test harness: a real fake-Ollama server with a real proxy in front of it.

Both run as actual HTTP servers so streaming behaviour is exercised for real,
not simulated by an in-process transport.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import StringIO

import pytest
import pytest_asyncio
import uvicorn
from rich.console import Console

from localdoctor.proxy import Settings, create_app
from localdoctor.store import Store

import tests.fake_ollama as fake


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def running(app, port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started and not task.done():
        await asyncio.sleep(0.01)
    if task.done():  # surfaced a startup error
        task.result()
    try:
        yield server
    finally:
        server.should_exit = True
        await task


@dataclass
class Stack:
    base_url: str
    upstream_url: str
    app: object
    store: Store
    output: StringIO

    def printed(self) -> str:
        return self.output.getvalue()

    async def settle(self) -> None:
        """Wait for the post-response analysis tasks to finish."""
        for _ in range(50):
            tasks = set(self.app.state.tasks)
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)
        raise AssertionError("analysis tasks never settled")


@pytest_asyncio.fixture
async def stack(tmp_path):
    fake.seen_requests.clear()
    fake.last_raw_body.clear()

    fake_port = free_port()
    proxy_port = free_port()
    upstream = f"http://127.0.0.1:{fake_port}"

    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100, no_color=True)
    store = Store(tmp_path / "test.db")
    app = create_app(Settings(upstream=upstream), store=store, console=console)

    async with running(fake.app, fake_port), running(app, proxy_port):
        yield Stack(
            base_url=f"http://127.0.0.1:{proxy_port}",
            upstream_url=upstream,
            app=app,
            store=store,
            output=output,
        )
    store.close()


@pytest_asyncio.fixture
async def upstream_down(tmp_path):
    """A proxy pointed at a port where nothing is listening."""
    dead_port = free_port()
    proxy_port = free_port()
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100, no_color=True)
    store = Store(tmp_path / "down.db")
    app = create_app(
        Settings(upstream=f"http://127.0.0.1:{dead_port}"), store=store, console=console
    )
    async with running(app, proxy_port):
        yield Stack(
            base_url=f"http://127.0.0.1:{proxy_port}",
            upstream_url=f"http://127.0.0.1:{dead_port}",
            app=app,
            store=store,
            output=output,
        )
    store.close()


CHAT = "/api/chat"


def chat_body(**overrides) -> dict:
    body = {
        "model": "fake:latest",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    body.update(overrides)
    return body


def make_record(**overrides):
    """A RequestRecord for unit-level rule tests, healthy unless overridden."""
    from localdoctor.models import ModelFacts, RequestRecord, Usage, new_id, now_iso

    facts = overrides.pop("model_facts", ModelFacts(name="m", context_length=32768, available=True))
    usage = overrides.pop(
        "usage", Usage(prompt_tokens=120, completion_tokens=9, finish_reason="stop")
    )
    base = dict(
        id=new_id(),
        ts=now_iso(),
        endpoint="/api/chat",
        request_body=b"{}",
        request_headers={},
        stream=False,
        model="m",
        status_code=200,
        request_json={"options": {"num_predict": 128}},
        prompt_text="hello there",
        output_text="a normal answer",
        num_ctx=8192,
        num_ctx_source="request",
    )
    base.update(overrides)
    return RequestRecord(usage=usage, model_facts=facts, **base)
