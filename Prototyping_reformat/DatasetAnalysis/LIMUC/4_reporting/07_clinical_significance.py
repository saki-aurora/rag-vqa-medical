#!/usr/bin/env python3
"""Compute remission-slice metrics and McNemar significance from full LIMUC runs."""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd

from _results_utils import collect_run_records, find_limuc_root, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to LIMUC root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output table folder. Default: <LIMUC>/4_reporting/results/tables",
    )
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=1686,
        help="Expected number of test rows for a full run.",
    )
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _remission_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    y_true_bin = (y_true <= 1).astype(int)
    y_pred_bin = (y_pred <= 1).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    n = int(len(y_true_bin))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, n)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "remission_accuracy": accuracy,
        "remission_precision": precision,
        "remission_recall_sensitivity": recall,
        "remission_specificity": specificity,
        "remission_f1": f1,
    }


def _mcnemar(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, float]:
    merged = df_a[["img_id", "y_true", "y_pred"]].rename(
        columns={"y_pred": "y_pred_a"}
    ).merge(
        df_b[["img_id", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_b"}),
        on=["img_id", "y_true"],
        how="inner",
    )
    y_true = pd.to_numeric(merged["y_true"], errors="coerce")
    y_a = pd.to_numeric(merged["y_pred_a"], errors="coerce")
    y_b = pd.to_numeric(merged["y_pred_b"], errors="coerce")
    valid = y_true.notna() & y_a.notna() & y_b.notna()
    y_true = y_true[valid].astype(int)
    y_a = y_a[valid].astype(int)
    y_b = y_b[valid].astype(int)

    a_correct = y_a == y_true
    b_correct = y_b == y_true
    n01 = int((~a_correct & b_correct).sum())  # A wrong, B right
    n10 = int((a_correct & ~b_correct).sum())  # A right, B wrong

    # McNemar with continuity correction.
    if (n01 + n10) > 0:
        chi2_cc = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
        # chi-square with df=1 -> survival function via erfc
        p_value = math.erfc(math.sqrt(max(chi2_cc, 0.0) / 2.0))
    else:
        chi2_cc = 0.0
        p_value = 1.0

    return {
        "n": int(len(y_true)),
        "n01_A_wrong_B_right": n01,
        "n10_A_right_B_wrong": n10,
        "chi2_cc": float(chi2_cc),
        "p_value": float(p_value),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_run_records(dataset_root=dataset_root, expected_test_rows=args.expected_test_rows)
    full_records = [r for r in records if int(r.get("is_full_run", 0)) == 1]
    if not full_records:
        raise RuntimeError("No full runs found; cannot compute remission/significance tables.")

    pred_by_run: Dict[str, pd.DataFrame] = {}
    remission_rows: List[Dict[str, float]] = []

    for r in full_records:
        run_name = str(r.get("run_name"))
        run_dir = dataset_root / str(r.get("run_dir"))
        pred_path = run_dir / "pred_test.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        if "img_id" not in pred.columns or "y_true" not in pred.columns or "y_pred" not in pred.columns:
            continue
        pred_by_run[run_name] = pred
        metrics = _remission_metrics(pred)
        remission_rows.append(
            {
                "run_name": run_name,
                "model": r.get("model"),
                "test_rows": r.get("test_rows"),
                **metrics,
                "run_dir": r.get("run_dir"),
            }
        )

    remission_rows.sort(key=lambda x: (-x["remission_accuracy"], x["run_name"]))
    remission_csv = output_dir / "chapter4_remission_slice_from_results.csv"
    write_csv(
        remission_rows,
        remission_csv,
        field_order=(
            "run_name",
            "model",
            "test_rows",
            "n",
            "remission_accuracy",
            "remission_precision",
            "remission_recall_sensitivity",
            "remission_specificity",
            "remission_f1",
            "tp",
            "tn",
            "fp",
            "fn",
            "run_dir",
        ),
    )

    mcnemar_rows: List[Dict[str, float]] = []
    run_names = sorted(pred_by_run.keys())
    for a, b in combinations(run_names, 2):
        stats = _mcnemar(pred_by_run[a], pred_by_run[b])
        mcnemar_rows.append(
            {
                "run_a": a,
                "run_b": b,
                "pair": f"{a} vs {b}",
                **stats,
            }
        )
    mcnemar_rows.sort(key=lambda x: x["p_value"])
    mcnemar_csv = output_dir / "chapter4_mcnemar_pairs_from_results.csv"
    write_csv(
        mcnemar_rows,
        mcnemar_csv,
        field_order=(
            "pair",
            "run_a",
            "run_b",
            "n",
            "n01_A_wrong_B_right",
            "n10_A_right_B_wrong",
            "chi2_cc",
            "p_value",
        ),
    )

    print(f"LIMUC root: {dataset_root}")
    print(f"Full runs used: {len(pred_by_run)}")
    print(f"Wrote remission table: {remission_csv}")
    print(f"Wrote McNemar table: {mcnemar_csv}")


if __name__ == "__main__":
    main()

