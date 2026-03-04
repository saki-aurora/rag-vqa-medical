#!/usr/bin/env python3
"""Evaluate PICO-driven retrieval against labeled relevant chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _parse_k_values(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    out = sorted({k for k in out if k >= 1})
    if not out:
        raise ValueError("k_values must contain at least one positive integer")
    return out


def parse_args() -> argparse.Namespace:
    root = _find_workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval_gold",
        type=Path,
        default=root / "data" / "queries" / "retrieval_gold.jsonl",
        help="Labeled relevant chunks per query.",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=root / "results" / "kb_build_latest" / "kb_manifest.json",
        help="Path to KB manifest.",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,3,5",
        help="Comma-separated k values for precision@k / recall@k.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "llm"],
        help="PICO extraction mode before retrieval.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "eval_latest",
        help="Output directory for retrieval_eval.json and per-query rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    workspace_root = _find_workspace_root()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    from pico_wrapper.pico_extract import extract_pico
    from pico_wrapper.retriever import retrieve_evidence
    from pico_wrapper.utils_io import ensure_dir, write_json, write_jsonl

    gold_path = args.retrieval_gold.resolve()
    manifest_path = args.manifest_path.resolve()
    if not gold_path.exists():
        raise FileNotFoundError(f"retrieval_gold not found: {gold_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path not found: {manifest_path}")

    rows = _read_jsonl(gold_path)
    if not rows:
        raise RuntimeError(f"No rows found in {gold_path}")
    k_values = _parse_k_values(args.k_values)
    max_k = max(k_values)

    out_dir = ensure_dir(args.out_dir.resolve())
    per_query_rows: List[Dict[str, object]] = []

    agg: Dict[int, Dict[str, float]] = {
        k: {"precision_sum": 0.0, "recall_sum": 0.0, "hit_sum": 0.0} for k in k_values
    }

    for row in rows:
        qid = str(row.get("qid", ""))
        query = str(row.get("query", "")).strip()
        relevant_chunk_ids = set(str(x) for x in row.get("relevant_chunk_ids", []))
        if not query:
            continue
        if not relevant_chunk_ids:
            continue

        pico = extract_pico(query=query, mode=args.mode)
        results = retrieve_evidence(
            query=query,
            manifest_path=manifest_path,
            pico=pico,
            k=max_k,
        )
        predicted = [r.chunk.chunk_id for r in results]

        row_metrics: Dict[str, Dict[str, float]] = {}
        for k in k_values:
            top_k = predicted[:k]
            hits = len(set(top_k).intersection(relevant_chunk_ids))
            p_at_k = _safe_div(hits, k)
            r_at_k = _safe_div(hits, len(relevant_chunk_ids))
            hit = 1.0 if hits > 0 else 0.0
            agg[k]["precision_sum"] += p_at_k
            agg[k]["recall_sum"] += r_at_k
            agg[k]["hit_sum"] += hit
            row_metrics[f"k={k}"] = {
                "hits": hits,
                "precision_at_k": p_at_k,
                "recall_at_k": r_at_k,
                "hit_at_k": hit,
            }

        per_query_rows.append(
            {
                "qid": qid,
                "query": query,
                "relevant_chunk_ids": sorted(relevant_chunk_ids),
                "predicted_chunk_ids": predicted,
                "metrics": row_metrics,
            }
        )

    n_eval = len(per_query_rows)
    if n_eval == 0:
        raise RuntimeError("No retrieval rows were evaluated (empty queries or no labels).")

    metrics: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        metrics[f"k={k}"] = {
            "precision_at_k": _safe_div(agg[k]["precision_sum"], n_eval),
            "recall_at_k": _safe_div(agg[k]["recall_sum"], n_eval),
            "hit_rate_at_k": _safe_div(agg[k]["hit_sum"], n_eval),
        }

    summary = {
        "task": "retrieval_eval",
        "mode": args.mode,
        "n_queries": n_eval,
        "k_values": k_values,
        "metrics": metrics,
        "input_path": str(gold_path),
        "manifest_path": str(manifest_path),
    }

    write_json(out_dir / "retrieval_eval.json", summary)
    write_jsonl(out_dir / "retrieval_eval_per_query.jsonl", per_query_rows)
    print(f"Retrieval eval complete for {n_eval} queries.")
    print(f"Output: {out_dir / 'retrieval_eval.json'}")


if __name__ == "__main__":
    main()
