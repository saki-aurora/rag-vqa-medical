"""Chapter 5 PICO wrapper package."""

from .schemas import (
    Citation,
    Claim,
    EvidenceChunk,
    PicoFrame,
    SeverityResult,
    WrapperInput,
    WrapperOutput,
)
from .safety import (
    DEFAULT_DISCLAIMER,
    contains_dosing_request,
    make_insufficient_evidence_message,
    make_standard_disclaimer,
    should_refuse_dosing,
)
from .utils_io import ensure_dir, freeze_environment, generate_run_id, read_json, write_json, write_jsonl
from .kb_ingest import build_kb_index
from .retriever import RetrievalResult, compose_pico_query, retrieve_evidence
from .pico_extract import extract_pico, extract_pico_baseline, llm_extractor_available
from .synthesis import synthesize_answer
from .wrapper import WrapperRunInfo, run_wrapper

__all__ = [
    "Citation",
    "Claim",
    "EvidenceChunk",
    "PicoFrame",
    "SeverityResult",
    "WrapperInput",
    "WrapperOutput",
    "DEFAULT_DISCLAIMER",
    "contains_dosing_request",
    "make_insufficient_evidence_message",
    "make_standard_disclaimer",
    "should_refuse_dosing",
    "ensure_dir",
    "freeze_environment",
    "generate_run_id",
    "read_json",
    "write_json",
    "write_jsonl",
    "build_kb_index",
    "RetrievalResult",
    "compose_pico_query",
    "retrieve_evidence",
    "extract_pico",
    "extract_pico_baseline",
    "llm_extractor_available",
    "synthesize_answer",
    "WrapperRunInfo",
    "run_wrapper",
]
