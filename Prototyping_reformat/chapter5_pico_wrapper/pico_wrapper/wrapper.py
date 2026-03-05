"""End-to-end wrapper orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .pico_extract import extract_pico, llm_extractor_available
from .retriever import RetrievalResult, retrieve_evidence
from .safety import (
    contains_contraindication_request,
    make_escalation_message,
    should_abstain_low_evidence,
    should_force_escalation,
)
from .schemas import SeverityResult, WrapperOutput
from .synthesis import synthesize_answer


@dataclass
class WrapperRunInfo:
    requested_mode: str
    used_mode: str
    query: str
    retrieval_k: int
    retrieval_backend: str
    rerank_enabled: bool
    rerank_pool: int
    rerank_alpha: float
    refusal_triggered: bool
    abstained_low_evidence: bool
    abstain_reason: str
    n_retrieved: int
    top_retrieval_score: float
    mean_retrieval_score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "used_mode": self.used_mode,
            "query": self.query,
            "retrieval_k": self.retrieval_k,
            "retrieval_backend": self.retrieval_backend,
            "rerank_enabled": self.rerank_enabled,
            "rerank_pool": self.rerank_pool,
            "rerank_alpha": self.rerank_alpha,
            "refusal_triggered": self.refusal_triggered,
            "abstained_low_evidence": self.abstained_low_evidence,
            "abstain_reason": self.abstain_reason,
            "n_retrieved": self.n_retrieved,
            "top_retrieval_score": self.top_retrieval_score,
            "mean_retrieval_score": self.mean_retrieval_score,
        }


def _resolve_mode(mode: str) -> str:
    mode = (mode or "baseline").strip().lower()
    if mode not in {"baseline", "llm"}:
        return "baseline"
    if mode == "llm" and not llm_extractor_available():
        return "baseline"
    return mode


def _normalize_severity(severity: Optional[SeverityResult | Dict[str, object]]) -> Optional[SeverityResult]:
    if severity is None:
        return None
    if isinstance(severity, SeverityResult):
        return severity
    if isinstance(severity, dict):
        return SeverityResult.from_dict(severity)
    raise TypeError("severity must be SeverityResult, dict, or None")


def run_wrapper(
    *,
    query: str,
    manifest_path: Path,
    run_id: str,
    retrieval_k: int = 5,
    retrieval_backend: Optional[str] = None,
    enable_rerank: bool = True,
    rerank_pool: int = 20,
    rerank_alpha: float = 0.20,
    min_top_score_for_answer: float = 0.18,
    min_mean_score_for_answer: float = 0.12,
    min_retrieved_for_answer: int = 2,
    mode: str = "baseline",
    severity: Optional[SeverityResult | Dict[str, object]] = None,
) -> tuple[WrapperOutput, WrapperRunInfo]:
    severity_obj = _normalize_severity(severity)
    requested_mode = (mode or "baseline").strip().lower()
    used_mode = _resolve_mode(requested_mode)

    pico = extract_pico(query, mode=used_mode)
    refusal = should_force_escalation(query)
    escalation_alert = make_escalation_message() if refusal else None

    retrieval_results: List[RetrievalResult] = retrieve_evidence(
        query=query,
        manifest_path=manifest_path,
        pico=pico,
        k=retrieval_k,
        backend_override=retrieval_backend,
        enable_rerank=enable_rerank,
        rerank_pool=rerank_pool,
        rerank_alpha=rerank_alpha,
    )
    retrieval_scores = [float(r.score) for r in retrieval_results]
    top_score = max(retrieval_scores) if retrieval_scores else 0.0
    mean_score = (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else 0.0
    abstain, abstain_reason = should_abstain_low_evidence(
        scores=retrieval_scores,
        min_top_score=min_top_score_for_answer,
        min_mean_score=min_mean_score_for_answer,
        min_results=min_retrieved_for_answer,
    )
    # Contraindication-focused queries get a stronger caution signal in limitations.
    if contains_contraindication_request(query) and escalation_alert is None:
        escalation_alert = "Contraindication/adverse-event context detected; require clinician verification."

    output = synthesize_answer(
        run_id=run_id,
        query=query,
        pico=pico,
        retrieval_results=retrieval_results,
        severity=severity_obj,
        refusal=refusal,
        abstain_low_evidence=abstain,
        abstain_reason=abstain_reason,
        escalation_alert=escalation_alert,
        mode=used_mode,
    )
    if requested_mode == "llm" and used_mode == "baseline":
        output.limitations.append(
            "Requested LLM mode but no local LLM backend was available; automatic fallback to baseline mode."
        )

    info = WrapperRunInfo(
        requested_mode=requested_mode,
        used_mode=used_mode,
        query=query,
        retrieval_k=retrieval_k,
        retrieval_backend=(retrieval_backend or "manifest_default"),
        rerank_enabled=bool(enable_rerank),
        rerank_pool=int(rerank_pool),
        rerank_alpha=float(rerank_alpha),
        refusal_triggered=refusal,
        abstained_low_evidence=abstain,
        abstain_reason=abstain_reason,
        n_retrieved=len(retrieval_results),
        top_retrieval_score=float(top_score),
        mean_retrieval_score=float(mean_score),
    )
    return output, info
