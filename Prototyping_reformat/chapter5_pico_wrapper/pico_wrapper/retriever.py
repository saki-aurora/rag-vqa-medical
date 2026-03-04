"""PICO-driven retrieval over persisted KB indexes."""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import EvidenceChunk, PicoFrame

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def compose_pico_query(query: str, pico: Optional[PicoFrame] = None) -> str:
    if pico is None:
        return query.strip()

    parts: List[str] = [query.strip()]
    if pico.population:
        parts.append("Population: " + ", ".join(pico.population))
    if pico.intervention:
        parts.append("Intervention: " + ", ".join(pico.intervention))
    if pico.comparator:
        parts.append("Comparator: " + ", ".join(pico.comparator))
    if pico.outcomes:
        parts.append("Outcomes: " + ", ".join(pico.outcomes))
    if pico.severity_anchors:
        parts.append("Severity: " + ", ".join(pico.severity_anchors))
    if pico.timeframe:
        parts.append("Timeframe: " + pico.timeframe)
    if pico.setting:
        parts.append("Setting: " + pico.setting)
    if pico.constraints:
        parts.append("Constraints: " + ", ".join(pico.constraints))
    return " | ".join([p for p in parts if p])


def _load_chunks(chunks_jsonl: Path) -> Dict[str, EvidenceChunk]:
    by_id: Dict[str, EvidenceChunk] = {}
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ch = EvidenceChunk.from_dict(obj)
            by_id[ch.chunk_id] = ch
    if not by_id:
        raise RuntimeError(f"No chunks loaded from {chunks_jsonl}")
    return by_id


@dataclass
class RetrievalResult:
    chunk: EvidenceChunk
    score: float
    rank: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk": self.chunk.to_dict(),
        }


class BaseRetriever:
    def retrieve(self, query: str, pico: Optional[PicoFrame] = None, k: int = 5) -> List[RetrievalResult]:
        raise NotImplementedError


class KeywordRetriever(BaseRetriever):
    def __init__(self, index_json: Path, chunks_by_id: Dict[str, EvidenceChunk]) -> None:
        data = json.loads(index_json.read_text(encoding="utf-8"))
        items = data.get("items", [])
        self._chunk_tokens: Dict[str, set[str]] = {}
        for item in items:
            cid = str(item["chunk_id"])
            toks = set(item.get("tokens", []))
            if cid in chunks_by_id:
                self._chunk_tokens[cid] = toks
        self._chunks_by_id = chunks_by_id

    @staticmethod
    def _score(query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(doc_tokens))
        denom = math.sqrt(len(query_tokens) * len(doc_tokens))
        return float(overlap / denom) if denom > 0 else 0.0

    def retrieve(self, query: str, pico: Optional[PicoFrame] = None, k: int = 5) -> List[RetrievalResult]:
        q = compose_pico_query(query, pico)
        q_tokens = set(_tokenize(q))
        scored: List[Tuple[str, float]] = []
        for cid, toks in self._chunk_tokens.items():
            score = self._score(q_tokens, toks)
            if score > 0:
                scored.append((cid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[RetrievalResult] = []
        for rank, (cid, score) in enumerate(scored[:k], start=1):
            out.append(
                RetrievalResult(
                    chunk=self._chunks_by_id[cid],
                    score=float(score),
                    rank=rank,
                )
            )
        return out


class TfidfRetriever(BaseRetriever):
    def __init__(self, index_pickle: Path, chunks_by_id: Dict[str, EvidenceChunk]) -> None:
        with index_pickle.open("rb") as f:
            payload = pickle.load(f)
        self._vectorizer = payload["vectorizer"]
        self._matrix = payload["matrix"]
        self._chunk_ids = payload["chunk_ids"]
        self._chunks_by_id = chunks_by_id

    def retrieve(self, query: str, pico: Optional[PicoFrame] = None, k: int = 5) -> List[RetrievalResult]:
        q = compose_pico_query(query, pico)
        q_vec = self._vectorizer.transform([q])
        # l2-normalized TF-IDF => dot product is cosine similarity
        scores = (self._matrix @ q_vec.T).toarray().ravel()
        idx = scores.argsort()[::-1]
        out: List[RetrievalResult] = []
        for rank, i in enumerate(idx[:k], start=1):
            score = float(scores[i])
            if score <= 0:
                continue
            cid = self._chunk_ids[int(i)]
            if cid not in self._chunks_by_id:
                continue
            out.append(
                RetrievalResult(
                    chunk=self._chunks_by_id[cid],
                    score=score,
                    rank=rank,
                )
            )
        return out


def load_retriever(index_dir: Path, chunks_jsonl: Path, backend: str) -> BaseRetriever:
    chunks_by_id = _load_chunks(chunks_jsonl)
    backend = backend.lower().strip()
    if backend == "tfidf":
        return TfidfRetriever(index_dir / "tfidf_index.pkl", chunks_by_id=chunks_by_id)
    if backend == "keyword":
        return KeywordRetriever(index_dir / "keyword_index.json", chunks_by_id=chunks_by_id)
    raise ValueError(f"Unsupported retrieval backend: {backend}")


def retrieve_evidence(
    query: str,
    *,
    manifest_path: Path,
    pico: Optional[PicoFrame] = None,
    k: int = 5,
) -> List[RetrievalResult]:
    if k < 1:
        raise ValueError("k must be >= 1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = manifest_path.parent
    index_dir = out_dir / "kb_index"
    chunks_jsonl = out_dir / "chunks.jsonl"
    backend = str(manifest.get("index_backend", "keyword"))
    retriever = load_retriever(index_dir=index_dir, chunks_jsonl=chunks_jsonl, backend=backend)
    return retriever.retrieve(query=query, pico=pico, k=k)
