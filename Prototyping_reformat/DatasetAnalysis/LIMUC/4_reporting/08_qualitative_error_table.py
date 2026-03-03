#!/usr/bin/env python3
"""Build qualitative error table comparing best supervised vs best generative runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from _results_utils import find_limuc_root, normalize_prediction_df, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to LIMUC root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--supervised-run",
        type=str,
        default="finetune_resnet50",
        help="Run name for supervised model (under */results/).",
    )
    parser.add_argument(
        "--generative-run",
        type=str,
        default="vlm_zero_shot_mayo",
        help="Run name for generative model (under */results/).",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=4,
        help="Target rows per qualitative group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
        raise FileNotFoundError(f"Run folder '{run_name}' not found under {dataset_root}")
    return matches[0]


def _sample_rows(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].copy()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sup_dir = _find_run_dir(dataset_root, args.supervised_run)
    gen_dir = _find_run_dir(dataset_root, args.generative_run)

    sup_pred_raw = pd.read_csv(sup_dir / "pred_test.csv")
    gen_pred_raw = pd.read_csv(gen_dir / "pred_test.csv")
    sup_pred = normalize_prediction_df(sup_pred_raw)
    gen_pred = normalize_prediction_df(gen_pred_raw)
    if sup_pred is None or gen_pred is None:
        raise RuntimeError("Prediction schema mismatch: required labels/predictions are missing.")

    gen_raw_path = gen_dir / "pred_test_raw.csv"
    gen_raw = normalize_prediction_df(pd.read_csv(gen_raw_path)) if gen_raw_path.exists() else None

    merge_cols = ["img_id_canonical", "y_true"]
    sup_cols = ["img_id_canonical", "img_id", "y_true", "y_pred"]
    if "image_path" in sup_pred.columns:
        sup_cols.append("image_path")
    sup = sup_pred[sup_cols].rename(
        columns={"y_pred": "y_pred_supervised"}
    )
    gen = gen_pred[["img_id_canonical", "img_id", "y_true", "y_pred"]].rename(
        columns={"y_pred": "y_pred_generative", "img_id": "img_id_generative"}
    )
    merged = sup.merge(gen, on=merge_cols, how="inner")
    if "img_id" not in merged.columns:
        merged["img_id"] = merged["img_id_canonical"]

    if gen_raw is not None and {"img_id_canonical", "y_true", "raw_text"}.issubset(gen_raw.columns):
        merged = merged.merge(
            gen_raw[["img_id_canonical", "y_true", "raw_text"]],
            on=["img_id_canonical", "y_true"],
            how="left",
        )
    elif "raw_text" in gen_pred.columns:
        merged = merged.merge(
            gen_pred[["img_id_canonical", "y_true", "raw_text"]],
            on=["img_id_canonical", "y_true"],
            how="left",
        )
    else:
        merged["raw_text"] = ""

    y_true = pd.to_numeric(merged["y_true"], errors="coerce")
    y_sup = pd.to_numeric(merged["y_pred_supervised"], errors="coerce")
    y_gen = pd.to_numeric(merged["y_pred_generative"], errors="coerce")
    valid = y_true.notna() & y_sup.notna() & y_gen.notna()
    merged = merged.loc[valid].copy()
    merged["y_true"] = y_true[valid].astype(int)
    merged["y_pred_supervised"] = y_sup[valid].astype(int)
    merged["y_pred_generative"] = y_gen[valid].astype(int)

    merged["sup_correct"] = merged["y_pred_supervised"] == merged["y_true"]
    merged["gen_correct"] = merged["y_pred_generative"] == merged["y_true"]

    rng = np.random.default_rng(args.seed)
    groups: Dict[str, pd.DataFrame] = {
        "correct_both": merged[merged["sup_correct"] & merged["gen_correct"]],
        "supervised_correct_generative_wrong": merged[merged["sup_correct"] & ~merged["gen_correct"]],
        "generative_correct_supervised_wrong": merged[~merged["sup_correct"] & merged["gen_correct"]],
    }

    sampled_parts: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, object]] = []
    for group_name, group_df in groups.items():
        sampled = _sample_rows(group_df, args.samples_per_group, rng)
        sampled = sampled.copy()
        sampled["category"] = group_name
        sampled_parts.append(sampled)
        coverage_rows.append(
            {
                "category": group_name,
                "available_rows": int(len(group_df)),
                "sampled_rows": int(len(sampled)),
            }
        )

    out_df = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else merged.iloc[0:0].copy()
    out_df["supervised_run"] = args.supervised_run
    out_df["generative_run"] = args.generative_run
    out_df["error_gap_supervised_minus_generative"] = (
        (out_df["y_pred_supervised"] - out_df["y_true"]).abs()
        - (out_df["y_pred_generative"] - out_df["y_true"]).abs()
    )

    out_cols = [
        "category",
        "img_id",
        "image_path",
        "y_true",
        "y_pred_supervised",
        "y_pred_generative",
        "sup_correct",
        "gen_correct",
        "error_gap_supervised_minus_generative",
        "raw_text",
        "supervised_run",
        "generative_run",
    ]
    qual_csv = output_dir / "chapter4_qualitative_error_table.csv"
    write_csv(out_df[out_cols].to_dict(orient="records"), qual_csv, field_order=out_cols)

    coverage_csv = output_dir / "chapter4_qualitative_error_table_coverage.csv"
    write_csv(
        coverage_rows,
        coverage_csv,
        field_order=("category", "available_rows", "sampled_rows"),
    )

    print(f"LIMUC root: {dataset_root}")
    print(f"Supervised run: {args.supervised_run}")
    print(f"Generative run: {args.generative_run}")
    print(f"Wrote qualitative table: {qual_csv}")
    print(f"Wrote coverage table: {coverage_csv}")


if __name__ == "__main__":
    main()
