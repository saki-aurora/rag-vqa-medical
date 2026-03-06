"""Safety policy helpers for Chapter 5 wrapper outputs."""

from __future__ import annotations

import re
from typing import Iterable, List


DEFAULT_DISCLAIMER = (
    "This output is decision-support and educational only, not medical advice. "
    "Clinical decisions require licensed clinician judgment and patient-specific review."
)

_DOSING_PATTERNS: List[str] = [
    r"\bdose\b",
    r"\bdosage\b",
    r"\bstart(?:ing)?\s+dose\b",
    r"\bmg\b",
    r"\bmilligram(?:s)?\b",
    r"\bmcg\b",
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b",
    r"\bhow much\b",
    r"\bhow many\b",
    r"\bprescribe\b",
    r"\bprescription\b",
    r"\b(?:once|twice)\s+daily\b",
    r"\b(?:bid|tid|qid|q\d+h)\b",
    r"\btitrat",
]

_EMERGENCY_PATTERNS: List[str] = [
    r"\bemergency\b",
    r"\burgent\b",
    r"\bsevere bleed",
    r"\bperforat",
    r"\btoxic megacolon",
    r"\bshock\b",
    r"\bicu\b",
]

_CONTRAINDICATION_PATTERNS: List[str] = [
    r"\bcontraindicat",
    r"\bdrug interaction",
    r"\badverse event",
    r"\bside effect",
]


def contains_dosing_request(query: str) -> bool:
    text = query.lower().strip()
    return any(re.search(p, text) for p in _DOSING_PATTERNS)


def should_refuse_dosing(query: str) -> bool:
    return contains_dosing_request(query)


def contains_emergency_request(query: str) -> bool:
    text = query.lower().strip()
    return any(re.search(p, text) for p in _EMERGENCY_PATTERNS)


def contains_contraindication_request(query: str) -> bool:
    text = query.lower().strip()
    return any(re.search(p, text) for p in _CONTRAINDICATION_PATTERNS)


def should_force_escalation(query: str) -> bool:
    return contains_dosing_request(query) or contains_emergency_request(query)


def make_standard_disclaimer() -> str:
    return DEFAULT_DISCLAIMER


def make_insufficient_evidence_message() -> str:
    return "Insufficient evidence in retrieved sources."


def make_escalation_message() -> str:
    return (
        "Potential high-risk or emergency context detected. Escalate immediately to urgent clinical evaluation "
        "and follow institutional emergency protocols."
    )


def should_abstain_low_evidence(
    *,
    scores: Iterable[float],
    min_top_score: float = 0.18,
    min_mean_score: float = 0.12,
    min_results: int = 2,
) -> tuple[bool, str]:
    vals = [float(s) for s in scores]
    if len(vals) < int(min_results):
        return True, f"retrieved_chunks<{int(min_results)}"
    top = max(vals) if vals else 0.0
    mean = sum(vals) / max(1, len(vals))
    if top < float(min_top_score):
        return True, f"top_score<{float(min_top_score):.3f}"
    if mean < float(min_mean_score):
        return True, f"mean_score<{float(min_mean_score):.3f}"
    return False, ""
