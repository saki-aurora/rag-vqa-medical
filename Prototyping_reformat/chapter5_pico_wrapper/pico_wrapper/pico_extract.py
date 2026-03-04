"""Rule-based PICO extraction with optional LLM-mode fallback."""

from __future__ import annotations

import os
import re
from typing import List

from .schemas import PicoFrame


POPULATION_PATTERNS = [
    r"\badults?\b",
    r"\bpediatric\b",
    r"\bchildren\b",
    r"\belderly\b",
    r"\bulcerative colitis\b",
    r"\buc\b",
    r"\bmoderate(?:\s+to\s+severe)?\b",
    r"\bsevere\b",
    r"\bmild\b",
]

INTERVENTION_TERMS = [
    "biologic",
    "advanced therapy",
    "anti-tnf",
    "infliximab",
    "adalimumab",
    "vedolizumab",
    "ustekinumab",
    "tofacitinib",
    "upadacitinib",
    "mesalamine",
    "steroid",
]

COMPARATOR_TERMS = [
    "placebo",
    "standard care",
    "usual care",
    "vs",
    "versus",
    "compared with",
]

OUTCOME_TERMS = [
    "remission",
    "clinical remission",
    "endoscopic improvement",
    "mucosal healing",
    "response",
    "adverse event",
    "hospitalization",
    "surgery",
]

SETTING_TERMS = [
    "inpatient",
    "outpatient",
    "emergency",
    "clinic",
    "hospital",
]

SEVERITY_PATTERNS = [
    r"\bmes\s*[:=]?\s*([0-3])\b",
    r"\bmayo(?:\s+endoscopic\s+subscore)?\s*[:=]?\s*([0-3])\b",
    r"\buceis\s*[:=]?\s*([0-8])\b",
    r"\bmild\b",
    r"\bmoderate\b",
    r"\bsevere\b",
    r"\bremission\b",
]

TIMEFRAME_PATTERNS = [
    r"\b\d{1,3}\s*-\s*\d{1,3}\s*(?:day|days|week|weeks|month|months|year|years)\b",
    r"\b\d{1,3}\s*(?:day|days|week|weeks|month|months|year|years)\b",
    r"\bshort[-\s]?term\b",
    r"\blong[-\s]?term\b",
]

CONSTRAINT_PATTERNS = [
    r"\bwithout\b[^.,;:!?]*",
    r"\bavoid\b[^.,;:!?]*",
    r"\bcontraindicat(?:ed|ion)\b[^.,;:!?]*",
    r"\bno\b\s+(?:dosing|dose|dosage)\b[^.,;:!?]*",
]


def _unique_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def _extract_patterns(text: str, patterns: List[str], normalized_label: str = "") -> List[str]:
    out: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if m.groups():
                val = m.group(1)
                if normalized_label:
                    out.append(f"{normalized_label} {val}")
                else:
                    out.append(m.group(0))
            else:
                out.append(m.group(0))
    return _unique_preserve(out)


def _extract_terms(text: str, terms: List[str]) -> List[str]:
    hits: List[str] = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            hits.append(term)
    return _unique_preserve(hits)


def _extract_comparators(text: str) -> List[str]:
    comps = _extract_terms(text, COMPARATOR_TERMS)
    for m in re.finditer(r"\b([a-z0-9\- ]{2,40})\s+(?:vs|versus)\s+([a-z0-9\- ]{2,40})\b", text, flags=re.IGNORECASE):
        lhs = " ".join(m.group(1).split())
        rhs = " ".join(m.group(2).split())
        if len(lhs.split()) <= 4 and len(rhs.split()) <= 4:
            comps.append(f"{lhs} vs {rhs}")
    return _unique_preserve(comps)


def _extract_timeframe(text: str) -> str | None:
    for pat in TIMEFRAME_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return " ".join(m.group(0).split())
    return None


def _extract_setting(text: str) -> str | None:
    hits = _extract_terms(text, SETTING_TERMS)
    return hits[0] if hits else None


def _extract_constraints(text: str) -> List[str]:
    out: List[str] = []
    for pat in CONSTRAINT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append(" ".join(m.group(0).split()))
    return _unique_preserve(out)


def extract_pico_baseline(query: str) -> PicoFrame:
    q = query.strip()
    population = _extract_patterns(q, POPULATION_PATTERNS)
    intervention = _extract_terms(q, INTERVENTION_TERMS)
    comparator = _extract_comparators(q)
    outcomes = _extract_terms(q, OUTCOME_TERMS)

    severity_anchors: List[str] = []
    severity_anchors.extend(_extract_patterns(q, [SEVERITY_PATTERNS[0], SEVERITY_PATTERNS[1]], normalized_label="MES"))
    severity_anchors.extend(_extract_patterns(q, [SEVERITY_PATTERNS[2]], normalized_label="UCEIS"))
    severity_anchors.extend(_extract_patterns(q, SEVERITY_PATTERNS[3:]))
    severity_anchors = _unique_preserve(severity_anchors)

    timeframe = _extract_timeframe(q)
    setting = _extract_setting(q)
    constraints = _extract_constraints(q)

    return PicoFrame(
        population=population,
        intervention=intervention,
        comparator=comparator,
        outcomes=outcomes,
        severity_anchors=severity_anchors,
        timeframe=timeframe,
        setting=setting,
        constraints=constraints,
    )


def llm_extractor_available() -> bool:
    # Plug-in hook: we only advertise availability when explicitly enabled.
    return os.getenv("CH5_LLM_ENABLED", "0").strip() == "1"


def extract_pico(query: str, mode: str = "baseline") -> PicoFrame:
    """Extract PICO with safe fallback.

    `mode='llm'` currently falls back to baseline unless `CH5_LLM_ENABLED=1`.
    """
    mode = (mode or "baseline").strip().lower()
    if mode == "llm" and llm_extractor_available():
        # LLM extractor hook intentionally deferred; baseline remains the safe default.
        return extract_pico_baseline(query)
    return extract_pico_baseline(query)
