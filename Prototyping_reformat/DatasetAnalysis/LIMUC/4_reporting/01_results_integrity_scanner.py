#!/usr/bin/env python3
"""Scan LIMUC results folders and export integrity tables for Chapter 4."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _results_utils import (
    DEFAULT_REQUIRED_ARTIFACTS,
    build_main_table_rows,
    build_missing_artifact_rows,
    collect_run_records,
    find_limuc_root,
    write_csv,
)


def _parse_required_artifacts(value: str) -> Sequence[str]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_expected_runs(value: str) -> Sequence[str]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


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
        help="Directory for CSV outputs. Default: <LIMUC>/4_reporting/results/tables",
    )
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=1686,
        help="Expected number of test rows for a full run.",
    )
    parser.add_argument(
        "--required-artifacts",
        type=_parse_required_artifacts,
        default=",".join(DEFAULT_REQUIRED_ARTIFACTS),
        help="Comma-separated required files for each run folder.",
    )
    parser.add_argument(
        "--expected-run-patterns",
        type=_parse_expected_runs,
        default=(
            "resnet50_frozen_logreg,"
            "vit_frozen_logreg,"
            "clip_linear_baseline,"
            "finetune_resnet50,"
            "finetune_vit_or_swin,"
            "vlm_zero_shot_mayo,"
            "vlm_lora_finetune_mayo"
        ),
        help=(
            "Comma-separated run-name patterns that should exist. "
            "Pattern match uses substring containment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    required_artifacts = args.required_artifacts

    run_records = collect_run_records(
        dataset_root=dataset_root,
        required_artifacts=required_artifacts,
        expected_test_rows=args.expected_test_rows,
    )
    if not run_records:
        raise RuntimeError(f"No run folders found under: {dataset_root}")

    full_runs = [r for r in run_records if int(r.get("is_full_run", 0)) == 1]
    smoke_runs = [r for r in run_records if int(r.get("is_smoke_or_subset", 0)) == 1]
    missing_rows = build_missing_artifact_rows(run_records, required_artifacts=required_artifacts)
    main_rows = build_main_table_rows(run_records, full_only=True)
    run_names = [str(r.get("run_name", "")) for r in run_records]
    full_run_names = [str(r.get("run_name", "")) for r in full_runs]
    missing_expected_rows = []
    for pattern in args.expected_run_patterns:
        found_any = any(pattern in run_name for run_name in run_names)
        found_full = any(pattern in run_name for run_name in full_run_names)
        if not found_any:
            status = "missing_all"
        elif not found_full:
            status = "present_only_as_smoke_or_subset"
        else:
            status = None
        if status is not None:
            missing_expected_rows.append(
                {
                    "run_pattern": pattern,
                    "status": status,
                }
            )

    write_csv(
        run_records,
        output_dir / "LIMUC_results_index.csv",
        field_order=(
            "track",
            "run_name",
            "run_dir",
            "test_rows",
            "val_rows",
            "expected_test_rows",
            "is_full_run",
            "is_smoke_or_subset",
            "missing_artifact_count",
            "missing_artifacts",
            "model",
            "model_name",
            "seed",
            "epochs",
            "lr",
            "split_hash",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "qwk",
            "mae",
            "rmse",
            "parse_rate",
        ),
    )
    write_csv(
        missing_rows,
        output_dir / "LIMUC_results_missing_artifacts.csv",
        field_order=("track", "run_name", "run_dir", "missing_artifact"),
    )
    write_csv(
        full_runs,
        output_dir / "LIMUC_results_summary_fullruns.csv",
        field_order=(
            "track",
            "run_name",
            "run_dir",
            "test_rows",
            "split_hash",
            "model",
            "model_name",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "qwk",
            "mae",
            "rmse",
            "parse_rate",
        ),
    )
    write_csv(
        main_rows,
        output_dir / "chapter4_main_table_from_results.csv",
        field_order=(
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
        ),
    )
    write_csv(
        missing_expected_rows,
        output_dir / "LIMUC_results_missing_expected_runs.csv",
        field_order=("run_pattern", "status"),
    )

    split_hashes = sorted({r.get("split_hash") for r in full_runs if r.get("split_hash")})
    print(f"Dataset root: {dataset_root}")
    print(f"Run folders scanned: {len(run_records)}")
    print(f"Full runs (test_rows={args.expected_test_rows}): {len(full_runs)}")
    print(f"Smoke/subset runs: {len(smoke_runs)}")
    print(f"Runs missing required artifacts: {len({r['run_name'] for r in missing_rows})}")
    print(f"Missing expected run patterns: {len(missing_expected_rows)}")
    print("Split hash by full run:")
    for row in sorted(full_runs, key=lambda x: str(x.get("run_name", ""))):
        print(f"  - {row.get('run_name')}: {row.get('split_hash')}")
    if len(split_hashes) == 1:
        print(f"Split-hash consistency: OK ({split_hashes[0]})")
    elif len(split_hashes) == 0:
        print("Split-hash consistency: UNKNOWN (no split_hash in full runs)")
    else:
        print(f"Split-hash consistency: MISMATCH ({len(split_hashes)} unique hashes)")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
