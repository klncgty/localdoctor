"""Token estimation: a lower bound, not a point estimate (spec 4.3).

We do not use `len(text) / 3.5`. It drifts badly on code, JSON, non-Latin
scripts and emoji, and drift produces false diagnoses. Instead we compute a
conservative LOWER BOUND: the real token count is almost certainly larger.

We never make a claim in the upper-bound direction.
"""

from __future__ import annotations

# Practical upper bound on how many characters a single BPE token can cover.
MAX_CHARS_PER_TOKEN = 12


def min_tokens(text: str) -> int:
    """Guaranteed lower bound on the token count of `text`.

    Isolated on purpose: if a real tokenizer is ever plugged in, only this
    function changes. Phase 1 ships no tokenizer — downloading an external
    model file would break the "fully local" principle.
    """
    if not text:
        return 0
    return len(text) // MAX_CHARS_PER_TOKEN
