"""End-to-end wrapper orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .pico_extract import extract_pico, llm_extractor_available
from .retriever import RetrievalResult, retrieve_evidence
from .safety import should_refuse_dosing
from .schemas import SeverityResult, WrapperOutput
from .synthesis import synthesize_answer


@dataclass
class WrapperRunInfo:
    requested_mode: str
    used_mode: str
    query: str
    retrieval_k: int
    refusal_triggered: bool
    n_retrieved: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "used_mode": self.used_mode,
            "query": self.query,
            "retrieval_k": self.retrieval_k,
            "refusal_triggered": self.refusal_triggered,
            "n_retrieved": self.n_retrieved,
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
    mode: str = "baseline",
    severity: Optional[SeverityResult | Dict[str, object]] = None,
) -> tuple[WrapperOutput, WrapperRunInfo]:
    severity_obj = _normalize_severity(severity)
    requested_mode = (mode or "baseline").strip().lower()
    used_mode = _resolve_mode(requested_mode)

    pico = extract_pico(query, mode=used_mode)
    refusal = should_refuse_dosing(query)

    retrieval_results: List[RetrievalResult] = retrieve_evidence(
        query=query,
        manifest_path=manifest_path,
        pico=pico,
        k=retrieval_k,
    )

    output = synthesize_answer(
        run_id=run_id,
        query=query,
        pico=pico,
        retrieval_results=retrieval_results,
        severity=severity_obj,
        refusal=refusal,
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
        refusal_triggered=refusal,
        n_retrieved=len(retrieval_results),
    )
    return output, info
