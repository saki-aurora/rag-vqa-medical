#!/usr/bin/env python3
"""Build a final Chapter 4 comparison table from LIMUC results folders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from _results_utils import (
    build_main_table_rows,
    collect_run_records,
    find_limuc_root,
    write_csv,
)


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
        help="Directory for table outputs. Default: <LIMUC>/4_reporting/results/tables",
    )
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=1686,
        help="Expected number of test rows for a full run.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke/subset runs in output (default: full runs only).",
    )
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(rows: Iterable[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records = collect_run_records(
        dataset_root=dataset_root,
        expected_test_rows=args.expected_test_rows,
    )
    if not run_records:
        raise RuntimeError(f"No run folders found under: {dataset_root}")

    if args.include_smoke:
        selected = build_main_table_rows(run_records, full_only=False)
    else:
        selected = build_main_table_rows(run_records, full_only=True)

    csv_columns = [
        "run_name",
        "track",
        "model",
        "model_name",
        "test_rows",
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
    write_csv(selected, output_dir / "chapter4_main_comparison_table.csv", field_order=csv_columns)

    md_columns = [
        "run_name",
        "model",
        "test_rows",
        "accuracy",
        "macro_f1",
        "balanced_acc",
        "qwk",
        "mae",
        "rmse",
        "parse_rate",
    ]
    md_text = []
    md_text.append("# Chapter 4 Main Comparison Table")
    md_text.append("")
    md_text.append(
        f"Source: `{dataset_root}`. "
        f"{'Includes smoke runs.' if args.include_smoke else 'Includes full runs only.'}"
    )
    md_text.append("")
    md_text.append(_to_markdown(selected, md_columns))
    md_text.append("")
    (output_dir / "chapter4_main_comparison_table.md").write_text(
        "\n".join(md_text),
        encoding="utf-8",
    )

    smoke_count = sum(int(r.get("is_smoke_or_subset", 0)) for r in run_records)
    print(f"Dataset root: {dataset_root}")
    print(f"Runs scanned: {len(run_records)}")
    print(f"Rows exported: {len(selected)}")
    print(f"Smoke/subset runs found: {smoke_count}")
    print(f"Wrote CSV: {output_dir / 'chapter4_main_comparison_table.csv'}")
    print(f"Wrote MD:  {output_dir / 'chapter4_main_comparison_table.md'}")


if __name__ == "__main__":
    main()

