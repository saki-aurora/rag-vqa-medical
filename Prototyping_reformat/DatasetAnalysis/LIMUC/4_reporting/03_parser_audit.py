#!/usr/bin/env python3
"""Create parser-audit samples for generative Mayo-score runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from _results_utils import find_limuc_root, write_csv


WORD_TO_MAYO = {
    "normal": 0,
    "remission": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


def parse_mayo(raw_text: Any) -> Optional[int]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None

    # Primary strict pattern used by chapter protocol.
    m = re.search(r"score\s*:\s*([0-3])", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Fallback: first explicit digit 0-3.
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return int(m.group(1))

    # Fallback: lexical severity words.
    lower = text.lower()
    for word, label in WORD_TO_MAYO.items():
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return label
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to LIMUC root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="vlm_zero_shot_mayo",
        help="Run folder name (under */results/) to audit.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Prediction split to audit.",
    )
    parser.add_argument(
        "--stratified-per-class",
        type=int,
        default=10,
        help="Target number of samples per true class (if available).",
    )
    parser.add_argument(
        "--random-n",
        type=int,
        default=20,
        help="Additional random samples across all classes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output table folder. Default: <LIMUC>/4_reporting/results/tables",
    )
    return parser.parse_args()


def _find_run_dir(dataset_root: Path, run_name: str) -> Path:
    matches = sorted(dataset_root.glob(f"**/results/{run_name}"))
    if not matches:
        raise FileNotFoundError(
            f"Run folder '{run_name}' not found under: {dataset_root}"
        )
    return matches[0]


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _find_run_dir(dataset_root, args.run_name)
    raw_csv = run_dir / f"pred_{args.split}_raw.csv"
    pred_csv = run_dir / f"pred_{args.split}.csv"
    source_csv = raw_csv if raw_csv.exists() else pred_csv
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing prediction file for split '{args.split}' in {run_dir}")

    df = pd.read_csv(source_csv)
    if df.empty:
        raise RuntimeError(f"Prediction file is empty: {source_csv}")

    true_col = _pick_col(df, ["y_true", "label_id"])
    pred_col = _pick_col(df, ["y_pred"])
    raw_col = _pick_col(df, ["raw_text"])
    id_col = _pick_col(df, ["img_id", "image_id"])

    if true_col is None or pred_col is None:
        raise RuntimeError(
            f"Required columns missing in {source_csv}. Found columns: {list(df.columns)}"
        )

    work = df.copy()
    work["true_label"] = pd.to_numeric(work[true_col], errors="coerce").astype("Int64")
    work["saved_pred"] = pd.to_numeric(work[pred_col], errors="coerce").astype("Int64")
    if id_col is not None:
        work["image_id"] = work[id_col].astype(str)
    else:
        # fallback to filename stem from image path
        if "image_path" in work.columns:
            work["image_id"] = work["image_path"].astype(str).apply(lambda p: Path(p).stem)
        else:
            work["image_id"] = work.index.astype(str)

    if raw_col is not None:
        work["raw_generation"] = work[raw_col].astype(str)
        work["parsed_label"] = work["raw_generation"].apply(parse_mayo).astype("Int64")
        work["parse_ok"] = work["parsed_label"].notna()
    else:
        work["raw_generation"] = ""
        work["parsed_label"] = work["saved_pred"]
        work["parse_ok"] = work["saved_pred"].isin([0, 1, 2, 3])

    work["parse_matches_saved_pred"] = work["parsed_label"] == work["saved_pred"]
    work["parse_matches_saved_pred"] = work["parse_matches_saved_pred"].fillna(False)

    rng = np.random.default_rng(args.seed)
    sampled_indices = set()

    # Stratified picks by true class.
    for cls in [0, 1, 2, 3]:
        class_idx = work.index[work["true_label"] == cls].to_list()
        if not class_idx:
            continue
        k = min(args.stratified_per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=k, replace=False).tolist()
        sampled_indices.update(chosen)

    # Add random picks from remaining rows.
    remaining = [idx for idx in work.index.to_list() if idx not in sampled_indices]
    if remaining and args.random_n > 0:
        k = min(args.random_n, len(remaining))
        chosen = rng.choice(remaining, size=k, replace=False).tolist()
        sampled_indices.update(chosen)

    sampled_df = work.loc[sorted(sampled_indices)].copy()
    sampled_df["source_run"] = args.run_name
    sampled_df["source_split"] = args.split
    sampled_df["source_file"] = str(source_csv.relative_to(dataset_root))

    out_cols = [
        "source_run",
        "source_split",
        "image_id",
        "true_label",
        "raw_generation",
        "parsed_label",
        "saved_pred",
        "parse_ok",
        "parse_matches_saved_pred",
        "source_file",
    ]
    sampled_rows = sampled_df[out_cols].to_dict(orient="records")

    out_csv = output_dir / f"parser_audit_samples_{args.run_name}_{args.split}.csv"
    write_csv(sampled_rows, out_csv, field_order=out_cols)

    summary = {
        "run_name": args.run_name,
        "split": args.split,
        "source_file": str(source_csv.relative_to(dataset_root)),
        "n_total_rows": int(len(work)),
        "n_sample_rows": int(len(sampled_df)),
        "parse_rate_full": float(work["parse_ok"].mean()),
        "parse_match_saved_pred_rate_full": float(work["parse_matches_saved_pred"].mean()),
        "n_parse_fail_full": int((~work["parse_ok"]).sum()),
        "n_parse_mismatch_full": int((~work["parse_matches_saved_pred"]).sum()),
    }
    out_json = output_dir / f"parser_audit_summary_{args.run_name}_{args.split}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run: {args.run_name}")
    print(f"Split: {args.split}")
    print(f"Source file: {source_csv}")
    print(f"Total rows: {len(work)}")
    print(f"Sample rows exported: {len(sampled_df)}")
    print(f"Parse rate (full set): {summary['parse_rate_full']:.4f}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")


if __name__ == "__main__":
    main()

