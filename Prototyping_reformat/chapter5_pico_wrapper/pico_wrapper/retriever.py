"""PICO-driven retrieval over persisted KB indexes."""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schemas import EvidenceChunk, PicoFrame

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_index_path(index_dir: Path, raw_path: object) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    # First try relative to kb_index/, then to the manifest directory.
    cand1 = (index_dir / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (index_dir.parent / p).resolve()
    if cand2.exists():
        return cand2
    return cand1


def _minmax_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = np.asarray(list(score_map.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return {k: 0.0 for k in score_map}
    if abs(vmax - vmin) <= 1e-12:
        if vmax <= 0:
            return {k: 0.0 for k in score_map}
        return {k: 1.0 for k in score_map}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in score_map.items()}


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
    backend: str = "unknown"
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "backend": self.backend,
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
            "rerank_score": self.rerank_score,
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
                    backend="keyword",
                    lexical_score=float(score),
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
        for i in idx:
            if len(out) >= k:
                break
            score = float(scores[i])
            if score <= 0:
                continue
            cid = self._chunk_ids[int(i)]
            if cid not in self._chunks_by_id:
                continue
            rank = len(out) + 1
            out.append(
                RetrievalResult(
                    chunk=self._chunks_by_id[cid],
                    score=score,
                    rank=rank,
                    backend="tfidf",
                    lexical_score=score,
                )
            )
        return out

    def score_map(self, query: str, pico: Optional[PicoFrame] = None) -> Dict[str, float]:
        q = compose_pico_query(query, pico)
        q_vec = self._vectorizer.transform([q])
        scores = (self._matrix @ q_vec.T).toarray().ravel()
        out: Dict[str, float] = {}
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            cid = self._chunk_ids[int(i)]
            if cid not in self._chunks_by_id:
                continue
            out[cid] = float(score)
        return out


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        tfidf_index: Path,
        keyword_index: Optional[Path],
        lsa_index: Optional[Path],
        chunks_by_id: Dict[str, EvidenceChunk],
        backend_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._tfidf = TfidfRetriever(tfidf_index, chunks_by_id=chunks_by_id)
        self._chunks_by_id = chunks_by_id
        self._keyword = None
        if keyword_index is not None and keyword_index.exists():
            self._keyword = KeywordRetriever(keyword_index, chunks_by_id=chunks_by_id)

        self._lsa_chunk_ids: Optional[List[str]] = None
        self._lsa_embeddings: Optional[np.ndarray] = None
        self._lsa_components: Optional[np.ndarray] = None
        if lsa_index is not None and lsa_index.exists():
            payload = np.load(lsa_index, allow_pickle=True)
            self._lsa_chunk_ids = [str(x) for x in payload["chunk_ids"].tolist()]
            self._lsa_embeddings = np.asarray(payload["embeddings"], dtype=float)
            self._lsa_components = np.asarray(payload["svd_components"], dtype=float)

        default_w = {"tfidf": 0.5, "keyword": 0.25, "semantic": 0.25}
        if backend_weights:
            for k, v in backend_weights.items():
                default_w[str(k)] = _safe_float(v, default=default_w.get(str(k), 0.0))
        total = sum(max(0.0, w) for w in default_w.values())
        if total <= 0:
            total = 1.0
        self._weights = {k: max(0.0, w) / total for k, w in default_w.items()}

    def _semantic_score_map(self, query: str, pico: Optional[PicoFrame]) -> Dict[str, float]:
        if self._lsa_chunk_ids is None or self._lsa_embeddings is None or self._lsa_components is None:
            return {}
        q = compose_pico_query(query, pico)
        q_vec = self._tfidf._vectorizer.transform([q])
        q_emb = q_vec @ self._lsa_components.T
        q_emb = np.asarray(q_emb, dtype=float).reshape(1, -1)
        q_norm = np.linalg.norm(q_emb)
        if q_norm <= 1e-12:
            return {}
        q_emb = q_emb / q_norm
        scores = (self._lsa_embeddings @ q_emb.T).ravel()
        out: Dict[str, float] = {}
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            cid = self._lsa_chunk_ids[int(i)]
            if cid in self._chunks_by_id:
                out[cid] = float(score)
        return out

    def retrieve(self, query: str, pico: Optional[PicoFrame] = None, k: int = 5) -> List[RetrievalResult]:
        tfidf_scores = self._tfidf.score_map(query, pico)
        keyword_scores: Dict[str, float] = {}
        if self._keyword is not None:
            q = compose_pico_query(query, pico)
            q_tokens = set(_tokenize(q))
            for cid, toks in self._keyword._chunk_tokens.items():
                s = self._keyword._score(q_tokens, toks)
                if s > 0:
                    keyword_scores[cid] = float(s)
        semantic_scores = self._semantic_score_map(query, pico)

        n_tfidf = _minmax_normalize(tfidf_scores)
        n_keyword = _minmax_normalize(keyword_scores)
        n_sem = _minmax_normalize(semantic_scores)

        candidates = set(n_tfidf.keys()) | set(n_keyword.keys()) | set(n_sem.keys())
        ranked: List[Tuple[str, float, float, float]] = []
        for cid in candidates:
            lt = float(n_tfidf.get(cid, 0.0))
            lk = float(n_keyword.get(cid, 0.0))
            ls = float(n_sem.get(cid, 0.0))
            lexical = self._weights.get("tfidf", 0.0) * lt + self._weights.get("keyword", 0.0) * lk
            semantic = self._weights.get("semantic", 0.0) * ls
            score = lexical + semantic
            if score > 0:
                ranked.append((cid, score, lexical, semantic))
        ranked.sort(key=lambda x: x[1], reverse=True)

        out: List[RetrievalResult] = []
        for rank, (cid, score, lexical, semantic) in enumerate(ranked[:k], start=1):
            out.append(
                RetrievalResult(
                    chunk=self._chunks_by_id[cid],
                    score=float(score),
                    rank=rank,
                    backend="hybrid",
                    lexical_score=float(lexical),
                    semantic_score=float(semantic),
                )
            )
        return out


def load_retriever(
    index_dir: Path,
    chunks_jsonl: Path,
    backend: str,
    *,
    manifest_payload: Optional[Dict[str, object]] = None,
) -> BaseRetriever:
    chunks_by_id = _load_chunks(chunks_jsonl)
    backend = backend.lower().strip()
    manifest_payload = manifest_payload or {}
    index_files = manifest_payload.get("index_files", {})
    if not isinstance(index_files, dict):
        index_files = {}
    backend_weights = manifest_payload.get("backend_weights", {})
    if not isinstance(backend_weights, dict):
        backend_weights = {}
    if backend == "tfidf":
        tfidf_index = _resolve_index_path(index_dir, index_files.get("tfidf", index_dir / "tfidf_index.pkl"))
        return TfidfRetriever(tfidf_index, chunks_by_id=chunks_by_id)
    if backend == "keyword":
        keyword_index = _resolve_index_path(index_dir, index_files.get("keyword", index_dir / "keyword_index.json"))
        return KeywordRetriever(keyword_index, chunks_by_id=chunks_by_id)
    if backend == "hybrid":
        tfidf_index = _resolve_index_path(index_dir, index_files.get("tfidf", index_dir / "tfidf_index.pkl"))
        keyword_index = _resolve_index_path(index_dir, index_files.get("keyword", index_dir / "keyword_index.json"))
        lsa_index = _resolve_index_path(index_dir, index_files.get("semantic_lsa", index_dir / "lsa_index.npz"))
        return HybridRetriever(
            tfidf_index=tfidf_index,
            keyword_index=keyword_index if keyword_index.exists() else None,
            lsa_index=lsa_index if lsa_index.exists() else None,
            chunks_by_id=chunks_by_id,
            backend_weights={str(k): _safe_float(v) for k, v in backend_weights.items()},
        )
    raise ValueError(f"Unsupported retrieval backend: {backend}")


def _rerank_query_tokens(query: str, pico: Optional[PicoFrame]) -> set[str]:
    text = compose_pico_query(query, pico)
    return set(_tokenize(text))


def _rerank_score(query_tokens: set[str], chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(chunk_text))
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(doc_tokens))
    coverage = overlap / max(1, len(query_tokens))
    precision = overlap / max(1, len(doc_tokens))
    return float(0.75 * coverage + 0.25 * precision)


def _apply_rerank(
    *,
    results: List[RetrievalResult],
    query: str,
    pico: Optional[PicoFrame],
    k: int,
    rerank_pool: int,
    rerank_alpha: float,
) -> List[RetrievalResult]:
    if not results:
        return []
    pool_n = max(k, min(len(results), max(1, rerank_pool)))
    pool = list(results[:pool_n])
    q_tokens = _rerank_query_tokens(query, pico)
    alpha = float(min(1.0, max(0.0, rerank_alpha)))
    for item in pool:
        rscore = _rerank_score(q_tokens, item.chunk.text)
        item.rerank_score = float(rscore)
        item.score = float((1.0 - alpha) * item.score + alpha * rscore)
    pool.sort(key=lambda x: x.score, reverse=True)
    for rank, item in enumerate(pool[:k], start=1):
        item.rank = rank
    return pool[:k]


def retrieve_evidence(
    query: str,
    *,
    manifest_path: Path,
    pico: Optional[PicoFrame] = None,
    k: int = 5,
    backend_override: Optional[str] = None,
    enable_rerank: bool = False,
    rerank_pool: int = 20,
    rerank_alpha: float = 0.20,
) -> List[RetrievalResult]:
    if k < 1:
        raise ValueError("k must be >= 1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = manifest_path.parent
    index_dir = out_dir / "kb_index"
    chunks_jsonl_raw = manifest.get("chunks_file", out_dir / "chunks.jsonl")
    chunks_jsonl = _resolve_index_path(index_dir, chunks_jsonl_raw)
    backend = str(backend_override or manifest.get("index_backend", "keyword")).strip().lower()
    retriever = load_retriever(
        index_dir=index_dir,
        chunks_jsonl=chunks_jsonl,
        backend=backend,
        manifest_payload=manifest,
    )
    fetch_k = max(int(k), int(rerank_pool)) if enable_rerank else int(k)
    base = retriever.retrieve(query=query, pico=pico, k=fetch_k)
    if not enable_rerank:
        return base[:k]
    return _apply_rerank(
        results=base,
        query=query,
        pico=pico,
        k=k,
        rerank_pool=rerank_pool,
        rerank_alpha=rerank_alpha,
    )
