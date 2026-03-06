"""KB ingestion and indexing utilities."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from .schemas import EvidenceChunk
from .utils_io import ensure_dir, write_json, write_jsonl

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    _HAS_SKLEARN = True
except Exception:
    TruncatedSVD = None
    TfidfVectorizer = None
    _HAS_SKLEARN = False


SUPPORTED_EXTENSIONS = {".md", ".txt"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
WORD_SPAN_RE = re.compile(r"\S+")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*$")


@dataclass
class ChunkingConfig:
    max_words: int = 180
    overlap_words: int = 30
    min_words: int = 30
    random_seed: int = 42

    def validate(self) -> None:
        if self.max_words < 20:
            raise ValueError("max_words must be >= 20")
        if self.overlap_words < 0:
            raise ValueError("overlap_words must be >= 0")
        if self.overlap_words >= self.max_words:
            raise ValueError("overlap_words must be < max_words")
        if self.min_words < 1:
            raise ValueError("min_words must be >= 1")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _slug_from_path(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    return "__".join(parts).lower().replace(" ", "_")


def iter_kb_files(kb_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(kb_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if p.name.lower() == "readme.md":
                continue
            files.append(p)
    return files


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _split_sections(text: str, suffix: str) -> List[Tuple[str, int, int]]:
    """Return list of (section_title, start_offset, end_offset)."""
    if suffix.lower() != ".md":
        return [("document", 0, len(text))]

    lines = text.splitlines(keepends=True)
    offsets: List[int] = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    headings: List[Tuple[str, int]] = []
    for idx, ln in enumerate(lines):
        m = HEADING_RE.match(ln)
        if m:
            headings.append((m.group("title").strip(), offsets[idx + 1]))

    if not headings:
        return [("document", 0, len(text))]

    sections: List[Tuple[str, int, int]] = []
    for i, (title, start) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else len(text)
        if end > start:
            sections.append((title, start, end))
    return sections or [("document", 0, len(text))]


def _chunk_text_spans(text: str, max_words: int, overlap_words: int, min_words: int) -> List[Tuple[int, int]]:
    spans = list(WORD_SPAN_RE.finditer(text))
    if not spans:
        return []

    out: List[Tuple[int, int]] = []
    step = max_words - overlap_words
    i = 0
    n = len(spans)
    while i < n:
        j = min(i + max_words, n)
        if (j - i) < min_words and out:
            break
        start = spans[i].start()
        end = spans[j - 1].end()
        out.append((start, end))
        if j >= n:
            break
        i += step
    return out


def _document_chunks(path: Path, kb_root: Path, cfg: ChunkingConfig) -> List[EvidenceChunk]:
    text = _read_text(path)
    rel = path.relative_to(kb_root)
    doc_id = _slug_from_path(rel)
    source_path = str(path)

    section_ranges = _split_sections(text, path.suffix.lower())
    chunks: List[EvidenceChunk] = []
    cidx = 1
    for section_title, sec_start, sec_end in section_ranges:
        sec_text = text[sec_start:sec_end]
        sec_spans = _chunk_text_spans(
            sec_text,
            max_words=cfg.max_words,
            overlap_words=cfg.overlap_words,
            min_words=cfg.min_words,
        )
        for local_start, local_end in sec_spans:
            global_start = sec_start + local_start
            global_end = sec_start + local_end
            ctext = text[global_start:global_end].strip()
            if not ctext:
                continue
            chunk_id = f"{doc_id}#c{cidx:04d}"
            chunks.append(
                EvidenceChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source_path=source_path,
                    text=ctext,
                    section=section_title,
                    start_offset=global_start,
                    end_offset=global_end,
                )
            )
            cidx += 1
    return chunks


def _persist_keyword_index(chunks: Sequence[EvidenceChunk], index_dir: Path) -> Path:
    payload: Dict[str, Any] = {"backend": "keyword", "items": []}
    for c in chunks:
        toks = sorted(set(_tokenize(c.text)))
        payload["items"].append(
            {
                "chunk_id": c.chunk_id,
                "tokens": toks,
            }
        )
    out_path = index_dir / "keyword_index.json"
    write_json(out_path, payload)
    return out_path


def _persist_tfidf_index(chunks: Sequence[EvidenceChunk], index_dir: Path) -> tuple[Path, Dict[str, Any]]:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b")
    matrix = vectorizer.fit_transform(texts)
    payload = {
        "backend": "tfidf",
        "chunk_ids": [c.chunk_id for c in chunks],
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    out_path = index_dir / "tfidf_index.pkl"
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return out_path, payload


def _persist_lsa_index(
    *,
    tfidf_payload: Dict[str, Any],
    index_dir: Path,
    random_seed: int,
    max_components: int = 256,
) -> Path | None:
    if not _HAS_SKLEARN or TruncatedSVD is None:
        return None
    matrix = tfidf_payload["matrix"]
    n_samples, n_features = matrix.shape
    max_rank = int(min(n_samples - 1, n_features - 1))
    if max_rank < 2:
        return None
    n_components = int(min(max_components, max_rank))
    if n_components < 2:
        return None

    svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
    dense = svd.fit_transform(matrix)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    dense = (dense / norms).astype(np.float32)

    out_path = index_dir / "lsa_index.npz"
    np.savez_compressed(
        out_path,
        chunk_ids=np.asarray(tfidf_payload["chunk_ids"], dtype=object),
        embeddings=dense,
        components=np.asarray(int(n_components)),
        svd_components=np.asarray(svd.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(svd.explained_variance_ratio_, dtype=np.float32),
    )
    return out_path


def _build_manifest(
    kb_dir: Path,
    chunks: Sequence[EvidenceChunk],
    source_files: Sequence[Path],
    cfg: ChunkingConfig,
    backend: str,
    index_file: Path,
    chunks_file: Path,
    *,
    available_backends: Sequence[str],
    backend_weights: Dict[str, float],
    index_files: Dict[str, str],
) -> Dict[str, Any]:
    docs = sorted({c.doc_id for c in chunks})
    sections = sorted({c.section for c in chunks if c.section})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kb_dir": str(kb_dir.resolve()),
        "n_source_files": len(source_files),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "doc_ids": docs,
        "sections": sections,
        "chunking": {
            "max_words": cfg.max_words,
            "overlap_words": cfg.overlap_words,
            "min_words": cfg.min_words,
            "seed": cfg.random_seed,
        },
        "index_backend": backend,
        "index_file": str(index_file.resolve()),
        "chunks_file": str(chunks_file.resolve()),
        "available_backends": list(available_backends),
        "index_files": dict(index_files),
        "backend_weights": dict(backend_weights),
        "reranker": {
            "enabled_default": True,
            "rerank_pool_default": 20,
            "rerank_alpha_default": 0.20,
        },
    }


def build_kb_index(
    kb_dir: Path,
    out_dir: Path,
    max_words: int = 180,
    overlap_words: int = 30,
    min_words: int = 30,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Build KB chunks + index and persist manifest artifacts.

    Returns summary with output paths.
    """
    kb_dir = kb_dir.resolve()
    out_dir = out_dir.resolve()
    index_dir = ensure_dir(out_dir / "kb_index")

    cfg = ChunkingConfig(
        max_words=max_words,
        overlap_words=overlap_words,
        min_words=min_words,
        random_seed=random_seed,
    )
    cfg.validate()

    source_files = iter_kb_files(kb_dir)
    if not source_files:
        raise RuntimeError(f"No KB source files found in {kb_dir} (expected .md/.txt)")

    chunks: List[EvidenceChunk] = []
    for src in source_files:
        chunks.extend(_document_chunks(src, kb_root=kb_dir, cfg=cfg))
    if not chunks:
        raise RuntimeError("No chunks were produced from KB input files.")

    chunks_jsonl = out_dir / "chunks.jsonl"
    write_jsonl(chunks_jsonl, [c.to_dict() for c in chunks])

    keyword_index = _persist_keyword_index(chunks, index_dir=index_dir)
    index_files: Dict[str, str] = {"keyword": str(keyword_index.resolve())}
    available_backends: List[str] = ["keyword"]
    backend_weights = {"keyword": 0.25, "tfidf": 0.5, "semantic": 0.25}

    tfidf_payload: Dict[str, Any] | None = None
    lsa_index: Path | None = None
    tfidf_index: Path | None = None
    if _HAS_SKLEARN:
        tfidf_index, tfidf_payload = _persist_tfidf_index(chunks, index_dir=index_dir)
        index_files["tfidf"] = str(tfidf_index.resolve())
        available_backends.append("tfidf")
        if tfidf_payload is not None:
            lsa_index = _persist_lsa_index(
                tfidf_payload=tfidf_payload,
                index_dir=index_dir,
                random_seed=cfg.random_seed,
            )
        if lsa_index is not None:
            index_files["semantic_lsa"] = str(lsa_index.resolve())
            available_backends.append("semantic_lsa")
            available_backends.append("hybrid")

    if "hybrid" in available_backends and tfidf_index is not None:
        backend = "hybrid"
        index_file = tfidf_index
    elif tfidf_index is not None:
        backend = "tfidf"
        index_file = tfidf_index
    else:
        backend = "keyword"
        index_file = keyword_index

    manifest = _build_manifest(
        kb_dir=kb_dir,
        chunks=chunks,
        source_files=source_files,
        cfg=cfg,
        backend=backend,
        index_file=index_file,
        chunks_file=chunks_jsonl,
        available_backends=available_backends,
        backend_weights=backend_weights,
        index_files=index_files,
    )
    manifest_path = out_dir / "kb_manifest.json"
    write_json(manifest_path, manifest)

    return {
        "kb_manifest_path": str(manifest_path),
        "chunks_jsonl_path": str(chunks_jsonl),
        "index_dir": str(index_dir),
        "index_backend": backend,
        "n_chunks": len(chunks),
        "n_docs": len({c.doc_id for c in chunks}),
        "n_source_files": len(source_files),
    }
