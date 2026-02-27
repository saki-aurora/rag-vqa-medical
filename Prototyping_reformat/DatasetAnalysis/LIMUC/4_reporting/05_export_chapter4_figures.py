#!/usr/bin/env python3
"""Export Chapter 4-ready figures from persisted LIMUC results folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
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
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for figure outputs. Default: <LIMUC>/4_reporting/results/figures",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Directory for helper table outputs. Default: <LIMUC>/4_reporting/results/tables",
    )
    parser.add_argument(
        "--best-supervised-run",
        type=str,
        default="finetune_resnet50",
        help="Run folder name for the best supervised model.",
    )
    parser.add_argument(
        "--best-generative-run",
        type=str,
        default="vlm_zero_shot_mayo",
        help="Run folder name for the best generative model.",
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
        help="Include smoke/subset generative runs in histogram exports.",
    )
    return parser.parse_args()


def _find_run_path(dataset_root: Path, run_name: str) -> Path | None:
    matches = sorted(dataset_root.glob(f"**/results/{run_name}"))
    if not matches:
        return None
    return matches[0]


def _copy_confusion(run_dir: Path, run_name: str, figures_dir: Path) -> bool:
    src = run_dir / "confusion_test.png"
    if not src.exists():
        return False
    dst = figures_dir / f"confusion_test_{run_name}.png"
    shutil.copy2(src, dst)
    return True


def _plot_class_distribution(dataset_root: Path, figures_dir: Path) -> bool:
    metadata_csv = dataset_root / "0_dataset_prep" / "out" / "metadata" / "metadata_enriched.csv"
    if not metadata_csv.exists():
        return False

    df = pd.read_csv(metadata_csv)
    if "split" not in df.columns:
        return False

    if "label_id" in df.columns:
        class_col = "label_id"
    elif "label_name" in df.columns:
        class_col = "label_name"
    else:
        return False

    pivot = (
        df.groupby(["split", class_col]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    split_order = [s for s in ["train", "val", "test"] if s in pivot.index]
    pivot = pivot.reindex(split_order)

    ax = pivot.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_title("LIMUC Class Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution_by_split.png", dpi=220)
    plt.close()
    return True


def _plot_pred_histogram(pred_csv: Path, run_name: str, figures_dir: Path) -> Dict[str, int] | None:
    if not pred_csv.exists():
        return None
    df = pd.read_csv(pred_csv)
    if "y_pred" not in df.columns:
        return None

    y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    counts = y_pred.value_counts(dropna=False).to_dict()
    class_counts = {int(c): int((y_pred == c).sum()) for c in [0, 1, 2, 3]}
    invalid_count = int((~y_pred.isin([0, 1, 2, 3])).sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [0, 1, 2, 3]
    ys = [class_counts[x] for x in xs]
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xlabel("Predicted Mayo score")
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted Label Distribution: {run_name}")
    if invalid_count > 0:
        ax.text(
            0.99,
            0.95,
            f"invalid={invalid_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(figures_dir / f"pred_label_histogram_{run_name}.png", dpi=220)
    plt.close(fig)

    output = {
        "run_name": run_name,
        "n_rows": int(len(df)),
        "pred_0": class_counts[0],
        "pred_1": class_counts[1],
        "pred_2": class_counts[2],
        "pred_3": class_counts[3],
        "pred_invalid": invalid_count,
    }
    if pd.notna(counts.get(float("nan"), None)):
        output["pred_nan"] = int(counts.get(float("nan"), 0))
    return output


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    figures_dir = (
        args.figures_dir.resolve()
        if args.figures_dir
        else (dataset_root / "4_reporting" / "results" / "figures")
    )
    tables_dir = (
        args.tables_dir.resolve()
        if args.tables_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    class_dist_ok = _plot_class_distribution(dataset_root, figures_dir)

    copied_confusions: List[str] = []
    for run_name in [args.best_supervised_run, args.best_generative_run]:
        run_dir = _find_run_path(dataset_root, run_name)
        if run_dir and _copy_confusion(run_dir, run_name, figures_dir):
            copied_confusions.append(run_name)

    run_records = collect_run_records(dataset_root=dataset_root, expected_test_rows=args.expected_test_rows)
    hist_rows: List[Dict[str, int]] = []
    for record in run_records:
        if record.get("track") != "3_vlm_severity":
            continue
        if not args.include_smoke and int(record.get("is_full_run", 0)) != 1:
            continue
        run_dir = dataset_root / str(record.get("run_dir"))
        pred_csv = run_dir / "pred_test.csv"
        hist_row = _plot_pred_histogram(pred_csv, str(record.get("run_name")), figures_dir)
        if hist_row:
            hist_rows.append(hist_row)

    write_csv(
        hist_rows,
        tables_dir / "pred_label_histogram_counts.csv",
        field_order=("run_name", "n_rows", "pred_0", "pred_1", "pred_2", "pred_3", "pred_invalid", "pred_nan"),
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Class distribution figure: {'written' if class_dist_ok else 'skipped (metadata missing)'}")
    print(f"Copied confusion figures for: {', '.join(copied_confusions) if copied_confusions else 'none'}")
    print(f"Generative histogram figures: {len(hist_rows)}")
    print(f"Wrote helper table: {tables_dir / 'pred_label_histogram_counts.csv'}")
    print(f"Figure directory: {figures_dir}")


if __name__ == "__main__":
    main()

