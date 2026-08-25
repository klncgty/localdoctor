"""/api/show wrapper + effective num_ctx resolution (spec 4.4, 4.5).

Note: this Ollama version's /api/show response has no `digest` field (only
/api/tags does). `modified_at` serves the same purpose as a fingerprint: it
changes when the model is re-pulled, which drops the cache entry.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from localdoctor.models import ModelFacts

TTL_SECONDS = 15 * 60
_PARAM_LINE = re.compile(r"^\s*(\S+)\s+(\S+)\s*$")


def _parse_parameters(text: Any) -> dict[str, str]:
    """`parameters` in /api/show is a newline-separated string, not a dict."""
    out: dict[str, str] = {}
    if not isinstance(text, str):
        return out
    for line in text.splitlines():
        match = _PARAM_LINE.match(line)
        if match:
            out[match.group(1)] = match.group(2).strip('"')
    return out


def _context_length(model_info: Any) -> int | None:
    """Find the `{architecture}.context_length` key inside model_info."""
    if not isinstance(model_info, dict):
        return None
    arch = model_info.get("general.architecture")
    if isinstance(arch, str):
        value = model_info.get(f"{arch}.context_length")
        if isinstance(value, int):
            return value
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def facts_from_show(name: str, payload: dict[str, Any]) -> ModelFacts:
    params = _parse_parameters(payload.get("parameters"))
    default_num_ctx = None
    if "num_ctx" in params:
        try:
            default_num_ctx = int(params["num_ctx"])
        except ValueError:
            default_num_ctx = None
    caps = payload.get("capabilities")
    return ModelFacts(
        name=name,
        context_length=_context_length(payload.get("model_info")),
        default_num_ctx=default_num_ctx,
        capabilities=tuple(caps) if isinstance(caps, list) else (),
        fingerprint=payload.get("modified_at") if isinstance(payload.get("modified_at"), str) else None,
        fetched_at=time.monotonic(),
        available=True,
    )


class ModelInfo:
    """Model info cache plus the observed-ceiling estimate.

    No module-level mutable state: the proxy owns a single instance.
    """

    def __init__(self, upstream: str, client: httpx.AsyncClient) -> None:
        self._upstream = upstream.rstrip("/")
        self._client = client
        self._cache: dict[str, ModelFacts] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Highest prompt_eval_count observed per model (spec 4.4, step 3).
        self._observed: dict[str, int] = {}

    def invalidate(self, model: str | None = None) -> None:
        if model is None:
            self._cache.clear()
        else:
            self._cache.pop(model, None)

    def note_observation(self, model: str | None, prompt_tokens: int | None) -> None:
        if model and isinstance(prompt_tokens, int) and prompt_tokens > 0:
            if prompt_tokens > self._observed.get(model, 0):
                self._observed[model] = prompt_tokens

    def observed_ceiling(self, model: str | None) -> int | None:
        return self._observed.get(model) if model else None

    async def get(self, model: str | None) -> ModelFacts | None:
        if not model:
            return None
        cached = self._cache.get(model)
        if cached and (time.monotonic() - cached.fetched_at) < TTL_SECONDS:
            return cached

        lock = self._locks.setdefault(model, asyncio.Lock())
        async with lock:
            cached = self._cache.get(model)
            if cached and (time.monotonic() - cached.fetched_at) < TTL_SECONDS:
                return cached
            facts = await self._fetch(model)
            if facts is None:
                # Unreachable: leave the unknown as unknown.
                facts = ModelFacts(name=model, fetched_at=time.monotonic(), available=False)
            elif cached and cached.fingerprint and facts.fingerprint != cached.fingerprint:
                # If the fingerprint changed, everything derived from it is stale.
                self._observed.pop(model, None)
            self._cache[model] = facts
            return facts

    async def _fetch(self, model: str) -> ModelFacts | None:
        try:
            response = await self._client.post(
                f"{self._upstream}/api/show", json={"model": model}, timeout=5.0
            )
            if response.status_code != 200:
                return None
            return facts_from_show(model, response.json())
        except Exception:
            return None

    # --- num_ctx resolution (spec 4.4) ----------------------------------

    def resolve_num_ctx(
        self, request_num_ctx: int | None, model: str | None, facts: ModelFacts | None
    ) -> tuple[int | None, str]:
        """In order: request -> model default -> observed -> unknown.

        Ollama may have a server-wide OLLAMA_CONTEXT_LENGTH that this proxy
        cannot see. That is why the third step is an ESTIMATE, and any diagnosis
        built on it may not rise above `medium`.
        """
        if request_num_ctx:
            return request_num_ctx, "request"
        if facts and facts.default_num_ctx:
            return facts.default_num_ctx, "model_default"
        observed = self.observed_ceiling(model)
        if observed:
            return observed, "observed"
        return None, "unknown"
