"""Reasoning block detectors (spec 6, R004).

No model-specific string matching. Instead a small detector list. The list is a
dict — not a plugin system.

Some models never expose their reasoning (hidden reasoning). In that case the
detector returns None and the rule is skipped silently. That is not an error and
is never reported as one.
"""

from __future__ import annotations

import re

from localdoctor.models import ReasoningSpan

_QWEN_THINK = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_GENERIC_XML = re.compile(r"<(thinking|reasoning)>(.*?)(?:</\1>|$)", re.DOTALL | re.IGNORECASE)


def _native_thinking_field(text: str, native_thinking: str) -> str | None:
    """Modern Ollama returns reasoning in a separate `message.thinking` field;
    no <think> tag ever appears in the content."""
    return native_thinking if native_thinking.strip() else None


def _qwen_think_tags(text: str, native_thinking: str) -> str | None:
    joined = "".join(_QWEN_THINK.findall(text))
    return joined if joined.strip() else None


def _generic_xml_tags(text: str, native_thinking: str) -> str | None:
    joined = "".join(m[1] for m in _GENERIC_XML.findall(text))
    return joined if joined.strip() else None


DETECTORS = {
    "native_thinking_field": _native_thinking_field,
    "qwen_think_tags": _qwen_think_tags,
    "generic_xml_tags": _generic_xml_tags,
}


def detect_reasoning(
    text: str, model_name: str | None, native_thinking: str = ""
) -> ReasoningSpan | None:
    for name, detector in DETECTORS.items():
        found = detector(text or "", native_thinking or "")
        if found:
            return ReasoningSpan(detector=name, text=found, char_len=len(found))
    return None


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks, leaving the final answer block behind."""
    return _GENERIC_XML.sub("", _QWEN_THINK.sub("", text or ""))
