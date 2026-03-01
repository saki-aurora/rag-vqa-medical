#!/usr/bin/env python3
"""Results-only Chapter 4 audit outputs requested for thesis completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_RUNS = [
    "finetune_resnet50",
    "finetune_vit_or_swin",
    "resnet50_frozen_logreg",
    "vit_frozen_logreg",
    "clip_linear_baseline",
    "vlm_zero_shot_mayo",
    "vlm_lora_finetune_mayo",
]


def _count_rows(csv_path: Path) -> int | None:
    if not csv_path.exists():
        return None
    try:
        # Row count excluding header.
        return max(sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1, 0)
    except Exception:
        return None


def _read_split_hash(run_dir: Path) -> str | None:
    run_meta = run_dir / "run_meta.json"
    if not run_meta.exists():
        return None
    try:
        meta = json.loads(run_meta.read_text(encoding="utf-8"))
        split_hash = meta.get("split_hash")
        return str(split_hash) if split_hash else None
    except Exception:
        return None


def _read_run_meta(run_dir: Path) -> Dict[str, object]:
    run_meta = run_dir / "run_meta.json"
    if not run_meta.exists():
        return {}
    try:
        data = json.loads(run_meta.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_metrics_summary(run_dir: Path) -> Dict[str, object]:
    p = run_dir / "metrics_test.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        return summary if isinstance(summary, dict) else {}
    except Exception:
        return {}


def _scan_runs(limuc_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    allowed_tracks = {"1_frozen_encoders", "2_supervised_finetuning", "3_vlm_severity"}
    for run_dir in sorted(limuc_root.glob("**/results/*")):
        if not run_dir.is_dir():
            continue
        rel = run_dir.relative_to(limuc_root)
        if not rel.parts or rel.parts[0] not in allowed_tracks:
            continue
        pred_test = run_dir / "pred_test.csv"
        pred_val = run_dir / "pred_val.csv"
        metrics_test = run_dir / "metrics_test.json"
        run_meta = run_dir / "run_meta.json"
        confusion_test_png = run_dir / "confusion_test.png"
        meta = _read_run_meta(run_dir)
        metrics = _read_metrics_summary(run_dir)
        rows.append(
            {
                "run_folder": run_dir.name,
                "path": str(run_dir.resolve()),
                "test_rows": _count_rows(pred_test),
                "split_hash": _read_split_hash(run_dir),
                "model": meta.get("model"),
                "model_name": meta.get("model_name"),
                "run_id": meta.get("run_id"),
                "timestamp_utc": meta.get("timestamp_utc"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_acc": metrics.get("balanced_accuracy", metrics.get("balanced_acc")),
                "qwk": metrics.get("qwk"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "parse_rate": metrics.get("parse_rate"),
                "has_metrics_test": metrics_test.exists(),
                "has_pred_test": pred_test.exists(),
                "has_pred_val": pred_val.exists(),
                "has_run_meta": run_meta.exists(),
                "has_confusion_png": confusion_test_png.exists(),
            }
        )
    return rows


def _detect_canonical_split_hash(index_df: pd.DataFrame) -> str | None:
    full = index_df[index_df["test_rows"] == 1686].copy()
    full = full[full["split_hash"].notna()]
    if full.empty:
        return None
    return full["split_hash"].value_counts().index[0]


def _write_missing_runs_md(index_df: pd.DataFrame, out_md: Path) -> None:
    full_df = index_df[index_df["test_rows"] == 1686].copy()
    present = set(full_df["run_folder"].astype(str).tolist())
    missing = [r for r in REQUIRED_RUNS if r not in present]

    lines: List[str] = []
    lines.append("# Chapter 4 Missing Runs")
    lines.append("")
    lines.append("Required full runs (`test_rows == 1686`) check:")
    lines.append("")
    for run_name in REQUIRED_RUNS:
        if run_name in present:
            lines.append(f"- [x] `{run_name}`")
        else:
            lines.append(f"- [ ] `{run_name}`")
    lines.append("")
    lines.append("## Missing Runs")
    lines.append("")
    if missing:
        for run_name in missing:
            lines.append(f"- `{run_name}`")
    else:
        lines.append("- None")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limuc-root",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC"),
        help="Path to LIMUC folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out"),
        help="Output directory for requested Chapter 4 audit files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _scan_runs(limuc_root)
    if not rows:
        raise RuntimeError(f"No run folders found under: {limuc_root}")

    index_df = pd.DataFrame(rows)
    index_csv = out_dir / "results_index.csv"
    index_df.to_csv(index_csv, index=False)

    canonical_split_hash = _detect_canonical_split_hash(index_df)
    main_df = index_df.copy()
    main_df = main_df[main_df["test_rows"] == 1686]
    if canonical_split_hash is not None:
        main_df = main_df[main_df["split_hash"] == canonical_split_hash]
    main_df = main_df.sort_values(["accuracy", "run_folder"], ascending=[False, True]).reset_index(drop=True)
    main_df = main_df[
        [
            "run_folder",
            "model",
            "model_name",
            "test_rows",
            "split_hash",
            "accuracy",
            "macro_f1",
            "balanced_acc",
            "qwk",
            "mae",
            "rmse",
            "parse_rate",
            "run_id",
            "timestamp_utc",
            "path",
        ]
    ]
    main_csv = out_dir / "chapter4_main_table_from_results.csv"
    main_df.to_csv(main_csv, index=False)

    missing_md = out_dir / "chapter4_missing_runs.md"
    _write_missing_runs_md(index_df, missing_md)

    print(f"LIMUC root: {limuc_root}")
    print(f"Run folders scanned: {len(index_df)}")
    print(f"Canonical split hash: {canonical_split_hash}")
    print(f"Wrote: {index_csv}")
    print(f"Wrote: {main_csv}")
    print(f"Wrote: {missing_md}")


if __name__ == "__main__":
    main()
