"""Deterministic synthesis with citation-linked claims."""

from __future__ import annotations

import re
from typing import List, Optional

from .retriever import RetrievalResult
from .safety import make_insufficient_evidence_message, make_standard_disclaimer
from .schemas import Citation, Claim, PicoFrame, SeverityResult, WrapperOutput


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if not parts:
        return text.strip()
    sentence = parts[0].strip()
    return sentence if sentence else text.strip()


def _severity_summary(severity: Optional[SeverityResult]) -> Optional[str]:
    if severity is None:
        return None
    conf = (
        f"{severity.confidence:.2f}" if severity.confidence is not None else "not reported"
    )
    probs = ""
    if severity.mes_probs:
        probs = (
            f" Probabilities: "
            f"[{severity.mes_probs[0]:.2f}, {severity.mes_probs[1]:.2f}, "
            f"{severity.mes_probs[2]:.2f}, {severity.mes_probs[3]:.2f}]."
        )
    cues = ""
    if severity.cues:
        cues = " Visual cues: " + "; ".join(severity.cues[:5]) + "."
    run = f" Source run: {severity.run_id}." if severity.run_id else ""
    return f"MES prediction: {severity.mes_pred} (confidence: {conf}).{probs}{cues}{run}".strip()


def _default_uncertainty(results: List[RetrievalResult], refusal: bool) -> str:
    if refusal:
        return "High uncertainty for patient-specific treatment execution due to safety constraints."
    if not results:
        return "High uncertainty: no supporting evidence chunks were retrieved."
    avg = sum(r.score for r in results) / len(results)
    if avg >= 0.35:
        return "Moderate uncertainty: evidence retrieval signal is reasonable but not definitive."
    if avg >= 0.2:
        return "Moderate-to-high uncertainty: evidence is partially relevant with limited strength."
    return "High uncertainty: weak lexical retrieval signal."


def _build_claims_and_citations(
    results: List[RetrievalResult],
    refusal: bool,
) -> tuple[List[Claim], List[Citation]]:
    citations: List[Citation] = []
    claims: List[Claim] = []

    for r in results:
        citations.append(
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                quote=_first_sentence(r.chunk.text),
            )
        )

    if refusal:
        cids = [c.chunk_id for c in citations[:2]]
        claims.append(
            Claim(
                text=(
                    "Specific patient-level dosing instructions are not provided by this wrapper. "
                    "Use retrieved evidence only for high-level decision-support discussion."
                ),
                citation_ids=cids,
            )
        )
        return claims, citations

    if not results:
        claims.append(Claim(text=make_insufficient_evidence_message(), citation_ids=[]))
        return claims, citations

    for r in results[:3]:
        claim_text = _first_sentence(r.chunk.text)
        claims.append(Claim(text=claim_text, citation_ids=[r.chunk.chunk_id]))
    return claims, citations


def synthesize_answer(
    *,
    run_id: str,
    query: str,
    pico: PicoFrame,
    retrieval_results: List[RetrievalResult],
    severity: Optional[SeverityResult] = None,
    refusal: bool = False,
    mode: str = "baseline",
) -> WrapperOutput:
    """Deterministic synthesis that enforces citation-linked claims."""
    mode = (mode or "baseline").strip().lower()
    limitations: List[str] = []

    if mode == "llm":
        limitations.append("LLM synthesis unavailable in this run; used deterministic baseline synthesis.")

    if severity is None:
        limitations.append("No severity context provided to wrapper input.")

    if not retrieval_results:
        limitations.append("No evidence chunks retrieved; response marked as insufficient evidence.")
    elif len(retrieval_results) < 3:
        limitations.append("Limited retrieval depth (<3 chunks); evidence coverage may be incomplete.")

    claims, citations = _build_claims_and_citations(retrieval_results, refusal=refusal)
    uncertainty = _default_uncertainty(retrieval_results, refusal=refusal)

    if refusal:
        limitations.append("Dosing-related request triggered safety refusal/escalation behavior.")

    return WrapperOutput(
        run_id=run_id,
        query=query,
        pico=pico,
        severity_summary=_severity_summary(severity),
        evidence=[r.chunk for r in retrieval_results],
        claims=claims,
        citations=citations,
        uncertainty=uncertainty,
        limitations=limitations or ["No major limitations reported."],
        disclaimer=make_standard_disclaimer(),
        refusal=refusal,
    )
