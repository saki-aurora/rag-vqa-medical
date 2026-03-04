#!/usr/bin/env python3
"""Evaluate PICO extraction against field-level gold annotations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")
FIELD_NAMES = [
    "population",
    "intervention",
    "comparator",
    "outcomes",
    "severity_anchors",
    "timeframe",
    "setting",
    "constraints",
]
REQUIRED_FIELDS = ["population", "intervention", "comparator", "outcomes", "severity_anchors"]


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


def _normalize_phrase(text: str) -> str:
    toks = TOKEN_RE.findall(text.lower())
    return " ".join(toks).strip()


def _normalize_values(raw_value: object) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        val = _normalize_phrase(raw_value)
        return [val] if val else []
    if isinstance(raw_value, list):
        out: List[str] = []
        for v in raw_value:
            if not isinstance(v, str):
                continue
            vv = _normalize_phrase(v)
            if vv:
                out.append(vv)
        return out
    return []


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def parse_args() -> argparse.Namespace:
    root = _find_workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pico_gold",
        type=Path,
        default=root / "data" / "queries" / "pico_gold.jsonl",
        help="Gold labels for PICO extraction.",
    )
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "llm"])
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "eval_latest",
        help="Output directory for pico_eval.json and per-query rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    workspace_root = _find_workspace_root()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    from pico_wrapper.pico_extract import extract_pico
    from pico_wrapper.utils_io import ensure_dir, write_json, write_jsonl

    gold_path = args.pico_gold.resolve()
    if not gold_path.exists():
        raise FileNotFoundError(f"pico_gold not found: {gold_path}")
    rows = _read_jsonl(gold_path)
    if not rows:
        raise RuntimeError(f"No rows found in {gold_path}")

    out_dir = ensure_dir(args.out_dir.resolve())
    per_query_rows: List[Dict[str, object]] = []

    counts: Dict[str, Dict[str, int]] = {
        field: {"tp": 0, "fp": 0, "fn": 0} for field in FIELD_NAMES
    }

    for row in rows:
        query = str(row.get("query", "")).strip()
        qid = str(row.get("qid", ""))
        gold_pico = row.get("pico", {})
        if not isinstance(gold_pico, dict):
            gold_pico = {}

        pred = extract_pico(query=query, mode=args.mode).to_dict()
        field_view: Dict[str, Dict[str, object]] = {}
        for field in FIELD_NAMES:
            pred_set = set(_normalize_values(pred.get(field)))
            gold_set = set(_normalize_values(gold_pico.get(field)))

            tp = len(pred_set.intersection(gold_set))
            fp = len(pred_set.difference(gold_set))
            fn = len(gold_set.difference(pred_set))
            counts[field]["tp"] += tp
            counts[field]["fp"] += fp
            counts[field]["fn"] += fn

            field_view[field] = {
                "pred": sorted(pred_set),
                "gold": sorted(gold_set),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        per_query_rows.append(
            {
                "qid": qid,
                "query": query,
                "mode": args.mode,
                "fields": field_view,
            }
        )

    per_field: Dict[str, Dict[str, float]] = {}
    for field in FIELD_NAMES:
        c = counts[field]
        m = _prf(c["tp"], c["fp"], c["fn"])
        per_field[field] = {
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        }

    required_f1 = [per_field[f]["f1"] for f in REQUIRED_FIELDS]
    macro_f1_required = sum(required_f1) / len(required_f1)
    macro_f1_all = sum(per_field[f]["f1"] for f in FIELD_NAMES) / len(FIELD_NAMES)

    summary = {
        "task": "pico_extraction_eval",
        "mode": args.mode,
        "n_queries": len(rows),
        "fields": per_field,
        "macro_f1_required_fields": macro_f1_required,
        "macro_f1_all_fields": macro_f1_all,
        "required_fields": REQUIRED_FIELDS,
        "field_order": FIELD_NAMES,
        "input_path": str(gold_path),
    }

    write_json(out_dir / "pico_eval.json", summary)
    write_jsonl(out_dir / "pico_eval_per_query.jsonl", per_query_rows)
    print(f"PICO eval complete for {len(rows)} queries.")
    print(f"Output: {out_dir / 'pico_eval.json'}")


if __name__ == "__main__":
    main()
