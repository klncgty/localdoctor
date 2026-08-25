"""Per-request capture and analysis orchestration.

The capture object is created per request and never shared — correlation is
done through this object, not through module-level state, so concurrent
requests cannot mix (spec 4.1).

Everything needed for replay (`localdoctor replay`, phase 3) is captured here:
the raw request body and headers are stored untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from localdoctor import normalize
from localdoctor.models import ChunkTiming, RequestRecord, Usage, new_id, now_iso
from localdoctor.modelinfo import ModelInfo
from localdoctor.report import Reporter
from localdoctor.rules.engine import Engine, EngineResult
from localdoctor.store import Store

# Upper bound on how much of a response we hold in memory for analysis.
# Beyond this we keep counting bytes but stop accumulating.
MAX_CAPTURE_BYTES = 16 * 1024 * 1024


@dataclass(slots=True)
class Capture:
    """Live state of one in-flight request."""

    id: str
    ts: str
    endpoint: str
    method: str
    request_body: bytes
    request_headers: dict[str, str]
    started: float = field(default_factory=time.monotonic)

    status_code: int | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    stream: bool = False

    ttft_ms: int | None = None
    total_ms: int | None = None
    chunk_count: int = 0
    chunks: list[ChunkTiming] = field(default_factory=list)

    buffer: bytearray = field(default_factory=bytearray)
    capture_truncated: bool = False
    upstream_error: str | None = None

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def note_chunk(self, data: bytes, record_chunks: bool) -> None:
        """Tee a chunk. Called while the chunk is already on its way out."""
        if not data:
            return
        offset = self.elapsed_ms()
        if self.ttft_ms is None:
            self.ttft_ms = offset
        self.chunk_count += 1
        if record_chunks:
            self.chunks.append(ChunkTiming(seq=self.chunk_count, offset_ms=offset, size=len(data)))
        if len(self.buffer) < MAX_CAPTURE_BYTES:
            self.buffer.extend(data)
        else:
            self.capture_truncated = True

    def finish(self) -> None:
        self.total_ms = self.elapsed_ms()


class Collector:
    """Turns a finished Capture into a RequestRecord, then diagnoses and stores it."""

    def __init__(
        self,
        model_info: ModelInfo,
        engine: Engine,
        reporter: Reporter,
        store: Store,
        record_chunks: bool = False,
    ) -> None:
        self.model_info = model_info
        self.engine = engine
        self.reporter = reporter
        self.store = store
        self.record_chunks = record_chunks

    def build_record(self, capture: Capture) -> RequestRecord:
        request_json = _safe_json(capture.request_body)
        raw_response = bytes(capture.buffer)
        decoded = normalize.maybe_decompress(
            raw_response, capture.response_headers.get("content-encoding")
        )
        parsed = normalize.parse_response(capture.endpoint, decoded, capture.stream)
        usage = normalize.normalize(request_json, parsed.meta, capture.endpoint)

        return RequestRecord(
            id=capture.id,
            ts=capture.ts,
            endpoint=capture.endpoint,
            request_body=capture.request_body,
            request_headers=capture.request_headers,
            stream=capture.stream,
            model=usage.model,
            response_body=raw_response or None,
            status_code=capture.status_code,
            ttft_ms=capture.ttft_ms,
            total_ms=capture.total_ms,
            chunk_count=capture.chunk_count,
            chunks=capture.chunks,
            usage=usage,
            request_json=request_json,
            output_text=parsed.content,
            thinking_text=parsed.thinking,
            prompt_text=normalize.extract_prompt_text(request_json, capture.endpoint),
            upstream_error=capture.upstream_error,
        )

    async def analyze(self, capture: Capture) -> RequestRecord:
        record = self.build_record(capture)
        record.model_facts = await self.model_info.get(record.model)

        # Resolve the window BEFORE recording this request's own token count,
        # otherwise the request would become its own observed ceiling and S1
        # would fire on every single call.
        record.num_ctx, record.num_ctx_source = self.model_info.resolve_num_ctx(
            record.usage.num_ctx_requested, record.model, record.model_facts
        )

        result = EngineResult()
        if record.status_code == 200 and not capture.capture_truncated:
            result = self.engine.run(record)
            self.reporter.emit(record, result)

        self.model_info.note_observation(record.model, record.usage.prompt_tokens)
        await self.store.awrite(record, result.all, self.record_chunks)
        return record


def _safe_json(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}
