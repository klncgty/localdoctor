"""Data models. Plain dataclasses — no ORM, no pydantic models."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Confidence = Literal["certain", "high", "medium", "low"]
Severity = Literal["error", "warning", "suggestion"]
NumCtxSource = Literal["request", "model_default", "observed", "unknown"]

# Lowest to highest. The one-step downgrade in R001 walks this order.
CONFIDENCE_ORDER: tuple[Confidence, ...] = ("low", "medium", "high", "certain")

# Lowest level that may reach the terminal. Spec 5: `low` is never printed.
PRINT_THRESHOLD: Confidence = "medium"

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id() -> str:
    """ULID-like 26-char id: lexicographic order matches chronological order."""
    n = (int(time.time() * 1000) << 80) | int.from_bytes(secrets.token_bytes(10), "big")
    out = []
    for _ in range(26):
        out.append(_B32[n & 31])
        n >>= 5
    return "".join(reversed(out))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def downgrade(confidence: Confidence, steps: int = 1) -> Confidence:
    """Lower a confidence by one (or n) step. Never goes below `low`."""
    idx = CONFIDENCE_ORDER.index(confidence)
    return CONFIDENCE_ORDER[max(0, idx - steps)]


def at_least(confidence: Confidence, floor: Confidence) -> bool:
    return CONFIDENCE_ORDER.index(confidence) >= CONFIDENCE_ORDER.index(floor)


@dataclass(slots=True)
class Usage:
    """Provider-independent usage figures (spec 4.2)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    model: str | None = None
    num_ctx_requested: int | None = None


@dataclass(slots=True)
class ReasoningSpan:
    """A reasoning block detected in the response (spec 6, R004)."""

    detector: str
    text: str
    char_len: int


@dataclass(slots=True)
class ChunkTiming:
    seq: int
    offset_ms: int
    size: int


@dataclass(slots=True)
class ModelFacts:
    """Model information derived from /api/show (spec 4.5)."""

    name: str
    context_length: int | None = None      # ceiling the architecture supports
    default_num_ctx: int | None = None     # modelfile PARAMETER num_ctx
    capabilities: tuple[str, ...] = ()
    fingerprint: str | None = None         # modified_at — /api/show has no digest
    fetched_at: float = 0.0
    available: bool = False                # did /api/show succeed


@dataclass(slots=True)
class RequestRecord:
    """Full record of one request. Replay-ready: raw body and headers kept."""

    id: str
    ts: str
    endpoint: str
    request_body: bytes
    request_headers: dict[str, str]
    stream: bool
    model: str | None = None
    response_body: bytes | None = None
    status_code: int | None = None
    ttft_ms: int | None = None
    total_ms: int | None = None
    chunk_count: int = 0
    chunks: list[ChunkTiming] = field(default_factory=list)

    # produced by normalize.py
    usage: Usage = field(default_factory=Usage)

    # resolved context window (spec 4.4)
    num_ctx: int | None = None
    num_ctx_source: NumCtxSource = "unknown"

    # derived fields the rule engine needs
    request_json: dict[str, Any] = field(default_factory=dict)
    output_text: str = ""        # final content returned to the user
    thinking_text: str = ""      # reasoning delivered in a separate field
    prompt_text: str = ""        # all input text extracted from the request
    model_facts: ModelFacts | None = None
    upstream_error: str | None = None

    @property
    def min_prompt_tokens(self) -> int:
        from localdoctor.estimate import min_tokens

        return min_tokens(self.prompt_text)


@dataclass(slots=True)
class Diagnosis:
    rule_id: str
    request_id: str
    confidence: Confidence
    severity: Severity
    title: str
    evidence: dict[str, Any]
    fix: str
    model: str | None = None
    ts: str = field(default_factory=now_iso)
    suppressed_by: str | None = None
    note: str | None = None
