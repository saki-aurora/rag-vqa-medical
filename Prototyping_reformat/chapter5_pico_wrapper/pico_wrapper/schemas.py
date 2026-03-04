"""Strict schemas for the Chapter 5 PICO wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _clean_str(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _clean_optional_str(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    cleaned = _clean_str(value, field_name)
    return cleaned


def _clean_str_list(values: Optional[List[str]], field_name: str) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list of strings")
    cleaned: List[str] = []
    for idx, v in enumerate(values):
        cleaned.append(_clean_str(v, f"{field_name}[{idx}]"))
    return cleaned


def _clean_float_01(value: Optional[float], field_name: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return out


@dataclass
class PicoFrame:
    population: List[str] = field(default_factory=list)
    intervention: List[str] = field(default_factory=list)
    comparator: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    severity_anchors: List[str] = field(default_factory=list)
    timeframe: Optional[str] = None
    setting: Optional[str] = None
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.population = _clean_str_list(self.population, "population")
        self.intervention = _clean_str_list(self.intervention, "intervention")
        self.comparator = _clean_str_list(self.comparator, "comparator")
        self.outcomes = _clean_str_list(self.outcomes, "outcomes")
        self.severity_anchors = _clean_str_list(self.severity_anchors, "severity_anchors")
        self.constraints = _clean_str_list(self.constraints, "constraints")
        self.timeframe = _clean_optional_str(self.timeframe, "timeframe")
        self.setting = _clean_optional_str(self.setting, "setting")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population": self.population,
            "intervention": self.intervention,
            "comparator": self.comparator,
            "outcomes": self.outcomes,
            "severity_anchors": self.severity_anchors,
            "timeframe": self.timeframe,
            "setting": self.setting,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PicoFrame":
        return cls(
            population=list(data.get("population", [])),
            intervention=list(data.get("intervention", [])),
            comparator=list(data.get("comparator", [])),
            outcomes=list(data.get("outcomes", [])),
            severity_anchors=list(data.get("severity_anchors", [])),
            timeframe=data.get("timeframe"),
            setting=data.get("setting"),
            constraints=list(data.get("constraints", [])),
        )


@dataclass
class SeverityResult:
    mes_pred: int
    mes_probs: Optional[List[float]] = None
    confidence: Optional[float] = None
    quality_flag: Optional[str] = None
    cues: List[str] = field(default_factory=list)
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mes_pred, int):
            raise TypeError("mes_pred must be int")
        if self.mes_pred < 0 or self.mes_pred > 3:
            raise ValueError("mes_pred must be in {0,1,2,3}")

        if self.mes_probs is not None:
            if not isinstance(self.mes_probs, list):
                raise TypeError("mes_probs must be a list of floats")
            if len(self.mes_probs) != 4:
                raise ValueError("mes_probs must have length 4")
            out: List[float] = []
            for idx, p in enumerate(self.mes_probs):
                fp = _clean_float_01(float(p), f"mes_probs[{idx}]")
                out.append(float(fp))
            self.mes_probs = out

        self.confidence = _clean_float_01(self.confidence, "confidence")
        self.quality_flag = _clean_optional_str(self.quality_flag, "quality_flag")
        self.cues = _clean_str_list(self.cues, "cues")
        self.model_version = _clean_optional_str(self.model_version, "model_version")
        self.run_id = _clean_optional_str(self.run_id, "run_id")
        self.timestamp = _clean_optional_str(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mes_pred": self.mes_pred,
            "mes_probs": self.mes_probs,
            "confidence": self.confidence,
            "quality_flag": self.quality_flag,
            "cues": self.cues,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeverityResult":
        return cls(
            mes_pred=int(data["mes_pred"]),
            mes_probs=data.get("mes_probs"),
            confidence=data.get("confidence"),
            quality_flag=data.get("quality_flag"),
            cues=list(data.get("cues", [])),
            model_version=data.get("model_version"),
            run_id=data.get("run_id"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class EvidenceChunk:
    doc_id: str
    chunk_id: str
    source_path: str
    text: str
    section: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    def __post_init__(self) -> None:
        self.doc_id = _clean_str(self.doc_id, "doc_id")
        self.chunk_id = _clean_str(self.chunk_id, "chunk_id")
        self.source_path = _clean_str(self.source_path, "source_path")
        self.text = _clean_str(self.text, "text")
        self.section = _clean_optional_str(self.section, "section")
        if self.start_offset is not None and self.start_offset < 0:
            raise ValueError("start_offset must be >= 0")
        if self.end_offset is not None and self.end_offset < 0:
            raise ValueError("end_offset must be >= 0")
        if self.start_offset is not None and self.end_offset is not None:
            if self.end_offset < self.start_offset:
                raise ValueError("end_offset must be >= start_offset")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "text": self.text,
            "section": self.section,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceChunk":
        return cls(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            source_path=data["source_path"],
            text=data["text"],
            section=data.get("section"),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
        )


@dataclass
class Citation:
    chunk_id: str
    doc_id: Optional[str] = None
    quote: Optional[str] = None

    def __post_init__(self) -> None:
        self.chunk_id = _clean_str(self.chunk_id, "chunk_id")
        self.doc_id = _clean_optional_str(self.doc_id, "doc_id")
        self.quote = _clean_optional_str(self.quote, "quote")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "quote": self.quote,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data.get("doc_id"),
            quote=data.get("quote"),
        )


@dataclass
class Claim:
    text: str
    citation_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = _clean_str(self.text, "text")
        self.citation_ids = _clean_str_list(self.citation_ids, "citation_ids")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "citation_ids": self.citation_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            text=data["text"],
            citation_ids=list(data.get("citation_ids", [])),
        )


@dataclass
class WrapperInput:
    query: str
    severity: Optional[SeverityResult] = None
    retrieval_k: int = 5
    mode: str = "baseline"

    def __post_init__(self) -> None:
        self.query = _clean_str(self.query, "query")
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")
        self.mode = _clean_str(self.mode, "mode").lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "severity": None if self.severity is None else self.severity.to_dict(),
            "retrieval_k": self.retrieval_k,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WrapperInput":
        sev = data.get("severity")
        return cls(
            query=data["query"],
            severity=None if sev is None else SeverityResult.from_dict(sev),
            retrieval_k=int(data.get("retrieval_k", 5)),
            mode=data.get("mode", "baseline"),
        )


@dataclass
class WrapperOutput:
    run_id: str
    query: str
    pico: PicoFrame
    severity_summary: Optional[str]
    evidence: List[EvidenceChunk]
    claims: List[Claim]
    citations: List[Citation]
    uncertainty: str
    limitations: List[str]
    disclaimer: str
    refusal: bool = False

    def __post_init__(self) -> None:
        self.run_id = _clean_str(self.run_id, "run_id")
        self.query = _clean_str(self.query, "query")
        if not isinstance(self.pico, PicoFrame):
            raise TypeError("pico must be PicoFrame")
        self.severity_summary = _clean_optional_str(self.severity_summary, "severity_summary")
        if not isinstance(self.evidence, list) or not all(isinstance(x, EvidenceChunk) for x in self.evidence):
            raise TypeError("evidence must be List[EvidenceChunk]")
        if not isinstance(self.claims, list) or not all(isinstance(x, Claim) for x in self.claims):
            raise TypeError("claims must be List[Claim]")
        if not isinstance(self.citations, list) or not all(isinstance(x, Citation) for x in self.citations):
            raise TypeError("citations must be List[Citation]")
        self.uncertainty = _clean_str(self.uncertainty, "uncertainty")
        self.limitations = _clean_str_list(self.limitations, "limitations")
        self.disclaimer = _clean_str(self.disclaimer, "disclaimer")
        if not isinstance(self.refusal, bool):
            raise TypeError("refusal must be bool")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "pico": self.pico.to_dict(),
            "severity_summary": self.severity_summary,
            "evidence": [x.to_dict() for x in self.evidence],
            "claims": [x.to_dict() for x in self.claims],
            "citations": [x.to_dict() for x in self.citations],
            "uncertainty": self.uncertainty,
            "limitations": self.limitations,
            "disclaimer": self.disclaimer,
            "refusal": self.refusal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WrapperOutput":
        return cls(
            run_id=data["run_id"],
            query=data["query"],
            pico=PicoFrame.from_dict(data["pico"]),
            severity_summary=data.get("severity_summary"),
            evidence=[EvidenceChunk.from_dict(x) for x in data.get("evidence", [])],
            claims=[Claim.from_dict(x) for x in data.get("claims", [])],
            citations=[Citation.from_dict(x) for x in data.get("citations", [])],
            uncertainty=data.get("uncertainty", "Uncertainty not provided."),
            limitations=list(data.get("limitations", [])),
            disclaimer=data.get("disclaimer", ""),
            refusal=bool(data.get("refusal", False)),
        )
