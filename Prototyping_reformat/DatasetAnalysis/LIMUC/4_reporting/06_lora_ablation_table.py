#!/usr/bin/env python3
"""Build a LoRA ablation summary table from persisted LIMUC results folders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    parser.add_argument(
        "--name-filter",
        type=str,
        default="lora",
        help="Substring used to identify LoRA run folders.",
    )
    return parser.parse_args()


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _to_md(rows: Iterable[Dict[str, Any]], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def _extract_lora_rank(row: Dict[str, Any]) -> Any:
    # Rank may be logged under different keys in future runs.
    for key in ("lora_r", "r", "rank", "lora_rank"):
        if key in row and row.get(key) is not None:
            return row.get(key)
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

    records = collect_run_records(dataset_root=dataset_root, expected_test_rows=args.expected_test_rows)
    lora_rows: List[Dict[str, Any]] = []
    for r in records:
        run_name = str(r.get("run_name", ""))
        model = str(r.get("model", ""))
        if args.name_filter.lower() not in run_name.lower() and args.name_filter.lower() not in model.lower():
            continue

        lora_rows.append(
            {
                "run_name": run_name,
                "track": r.get("track"),
                "test_rows": r.get("test_rows"),
                "is_full_run": r.get("is_full_run"),
                "model": r.get("model"),
                "model_name": r.get("model_name"),
                "seed": r.get("seed"),
                "epochs": r.get("epochs"),
                "lr": r.get("lr"),
                "lora_rank": _extract_lora_rank(r),
                "accuracy": r.get("accuracy"),
                "macro_f1": r.get("macro_f1"),
                "balanced_acc": r.get("balanced_accuracy"),
                "qwk": r.get("qwk"),
                "mae": r.get("mae"),
                "rmse": r.get("rmse"),
                "parse_rate": r.get("parse_rate"),
                "split_hash": r.get("split_hash"),
                "run_dir": r.get("run_dir"),
            }
        )

    lora_rows.sort(
        key=lambda row: (
            -1 if row.get("is_full_run") else 0,
            -1.0 if row.get("accuracy") is None else -float(row["accuracy"]),
            str(row.get("run_name", "")),
        )
    )

    csv_cols = [
        "run_name",
        "track",
        "test_rows",
        "is_full_run",
        "model",
        "model_name",
        "seed",
        "epochs",
        "lr",
        "lora_rank",
        "accuracy",
        "macro_f1",
        "balanced_acc",
        "qwk",
        "mae",
        "rmse",
        "parse_rate",
        "split_hash",
        "run_dir",
    ]
    out_csv = output_dir / "chapter4_lora_ablation_table.csv"
    write_csv(lora_rows, out_csv, field_order=csv_cols)

    md_cols = [
        "run_name",
        "test_rows",
        "is_full_run",
        "epochs",
        "lr",
        "lora_rank",
        "accuracy",
        "macro_f1",
        "balanced_acc",
        "qwk",
        "parse_rate",
    ]
    out_md = output_dir / "chapter4_lora_ablation_table.md"
    md = []
    md.append("# Chapter 4 LoRA Ablation Table")
    md.append("")
    md.append(f"Source root: `{dataset_root}`")
    md.append("")
    if lora_rows:
        md.append(_to_md(lora_rows, md_cols))
    else:
        md.append("No LoRA runs found in results folders.")
    md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")

    full_count = sum(int(row.get("is_full_run", 0)) for row in lora_rows)
    print(f"LIMUC root: {dataset_root}")
    print(f"LoRA runs found: {len(lora_rows)}")
    print(f"LoRA full runs (n={args.expected_test_rows}): {full_count}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD:  {out_md}")


if __name__ == "__main__":
    main()

