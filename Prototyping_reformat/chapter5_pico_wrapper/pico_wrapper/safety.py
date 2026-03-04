"""Safety policy helpers for Chapter 5 wrapper outputs."""

from __future__ import annotations

import re
from typing import List


DEFAULT_DISCLAIMER = (
    "This output is decision-support and educational only, not medical advice. "
    "Clinical decisions require licensed clinician judgment and patient-specific review."
)

_DOSING_PATTERNS: List[str] = [
    r"\bdose\b",
    r"\bdosage\b",
    r"\bmg\b",
    r"\bhow much\b",
    r"\bprescribe\b",
    r"\bprescription\b",
    r"\btitrat",
]


def contains_dosing_request(query: str) -> bool:
    text = query.lower().strip()
    return any(re.search(p, text) for p in _DOSING_PATTERNS)


def should_refuse_dosing(query: str) -> bool:
    return contains_dosing_request(query)


def make_standard_disclaimer() -> str:
    return DEFAULT_DISCLAIMER


def make_insufficient_evidence_message() -> str:
    return "Insufficient evidence in retrieved sources."

