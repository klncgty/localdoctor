"""FastAPI app: streaming passthrough, routing, tee (spec 4.1).

Two rules dominate this file:

1. Never modify the request or the response. Bytes go through untouched.
2. Never buffer a stream before forwarding it. Chunks are handed to the client
   the moment they arrive, and tee'd at the same time. Getting this wrong makes
   the product unusable.

Anything that does not match an analyzed endpoint is dumb passthrough. The
default is passthrough, not a whitelist.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from rich.console import Console

from localdoctor.collector import Capture, Collector
from localdoctor.models import new_id, now_iso
from localdoctor.modelinfo import ModelInfo
from localdoctor.normalize import ANALYZED_PATHS, CACHE_MUTATING_PATHS
from localdoctor.report import Reporter
from localdoctor.rules.engine import Engine
from localdoctor.store import DEFAULT_DB, Store

# Headers that belong to a single hop and must not be forwarded.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

PASSTHROUGH_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


@dataclass
class Settings:
    upstream: str = "http://localhost:11434"
    quiet: bool = False
    record_chunks: bool = False
    db_path: Path | str = DEFAULT_DB


def _forward_request_headers(request: Request) -> dict[str, str]:
    """Client headers minus hop-by-hop. Host and Content-Length are set by httpx."""
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP and key.lower() not in ("host", "content-length")
    }


def _forward_response_headers(response: httpx.Response) -> dict[str, str]:
    """Upstream headers minus hop-by-hop. content-encoding is kept: we forward
    the still-encoded bytes, so the client sees exactly what upstream sent."""
    return {
        key: value for key, value in response.headers.items() if key.lower() not in HOP_BY_HOP
    }


def _is_stream_response(response: httpx.Response, request_json: dict) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    if "event-stream" in content_type or "x-ndjson" in content_type:
        return True
    if "application/json" in content_type:
        return False
    value = request_json.get("stream")
    return bool(value)


def create_app(
    settings: Settings, store: Store | None = None, console: Console | None = None
) -> FastAPI:
    upstream = settings.upstream.rstrip("/")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # No read timeout: a long generation is not a failure.
        timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        app.state.client = client
        app.state.store = store or Store(settings.db_path)
        app.state.model_info = ModelInfo(upstream, client)
        app.state.collector = Collector(
            model_info=app.state.model_info,
            engine=Engine(),
            reporter=Reporter(console=console, quiet=settings.quiet),
            store=app.state.store,
            record_chunks=settings.record_chunks,
        )
        # Keep strong references so background analysis tasks are not collected.
        app.state.tasks = set()
        try:
            yield
        finally:
            for task in list(app.state.tasks):
                task.cancel()
            await client.aclose()
            if store is None:
                app.state.store.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/{full_path:path}", methods=PASSTHROUGH_METHODS)
    async def gateway(full_path: str, request: Request) -> Response:
        path = "/" + full_path
        body = await request.body()

        if request.method == "POST" and path in ANALYZED_PATHS:
            return await _proxy(request, path, body, analyze=True)
        if path in CACHE_MUTATING_PATHS:
            # Observed even on passthrough: these change what /api/show returns.
            request.app.state.model_info.invalidate(_model_name(body))
        return await _proxy(request, path, body, analyze=False)

    async def _proxy(request: Request, path: str, body: bytes, analyze: bool) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        url = httpx.URL(upstream + path, query=request.url.query.encode())
        # Build the httpx.Request directly so no client-default header (such as
        # accept-encoding) is silently added to what the caller sent.
        outgoing = httpx.Request(
            request.method, url, headers=_forward_request_headers(request), content=body
        )

        capture: Capture | None = None
        if analyze:
            capture = Capture(
                id=new_id(),
                ts=now_iso(),
                endpoint=path,
                method=request.method,
                request_body=body,
                request_headers=dict(request.headers),
            )

        try:
            response = await client.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            # Upstream is down. Report it as-is; do not stack our own diagnosis
            # on top of it.
            if capture is not None:
                capture.upstream_error = f"{type(exc).__name__}: {exc}"
                capture.status_code = 502
                capture.finish()
                _schedule(request.app, capture)
            return JSONResponse(
                {"error": f"localdoctor: upstream {upstream} unreachable ({type(exc).__name__})"},
                status_code=502,
            )

        if capture is not None:
            capture.status_code = response.status_code
            capture.response_headers = dict(response.headers)
            capture.stream = _is_stream_response(response, _safe_json(body))

        record_chunks = settings.record_chunks

        async def body_iterator():
            try:
                async for chunk in response.aiter_raw():
                    if capture is not None:
                        capture.note_chunk(chunk, record_chunks)
                    yield chunk
            finally:
                await response.aclose()
                if capture is not None:
                    capture.finish()
                    _schedule(request.app, capture)

        return StreamingResponse(
            body_iterator(),
            status_code=response.status_code,
            headers=_forward_response_headers(response),
        )

    return app


def _schedule(app: FastAPI, capture: Capture) -> None:
    """Run analysis after the response is done, never in the client's path."""
    collector: Collector = app.state.collector
    task = asyncio.create_task(collector.analyze(capture))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)


def _safe_json(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _model_name(raw: bytes) -> str | None:
    body = _safe_json(raw)
    for key in ("model", "name", "destination"):
        value = body.get(key)
        if isinstance(value, str):
            return value
    return None
