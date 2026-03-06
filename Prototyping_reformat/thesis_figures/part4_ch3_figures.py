#!/usr/bin/env python3
"""Part 4: Build Chapter 3 figure data and plots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent
OUT_DIR = WORKSPACE_DIR / "out"
DATA_DIR = WORKSPACE_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FREEZE_MANIFEST_PATH = OUT_DIR / "freeze_manifest.json"
PART4_SUMMARY_PATH = OUT_DIR / "part4_ch3_summary.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_freeze_index() -> Dict[str, Path]:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing freeze manifest: {FREEZE_MANIFEST_PATH}. Run freeze_inputs.py first."
        )
    manifest = _read_json(FREEZE_MANIFEST_PATH)
    by_id: Dict[str, Path] = {}
    for item in manifest.get("inputs", []):
        input_id = str(item.get("input_id", ""))
        abs_path = Path(str(item.get("abs_path", "")))
        exists = bool(item.get("exists", False))
        if exists:
            by_id[input_id] = abs_path
    return by_id


def _as_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return default
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return default


def _norm(value: object) -> str:
    return str(value).strip().lower()


def _parse_markdown_table(lines: List[str]) -> pd.DataFrame:
    if len(lines) < 2:
        return pd.DataFrame()
    headers = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    data_rows: List[Dict[str, object]] = []
    for line in lines[2:]:
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]
        data_rows.append(dict(zip(headers, cells)))
    return pd.DataFrame(data_rows)


def _table_after_heading(text: str, heading: str) -> pd.DataFrame:
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Heading not found: {heading}")

    table_start = None
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip().startswith("|"):
            table_start = idx
            break
    if table_start is None:
        raise ValueError(f"No table found after heading: {heading}")

    table_lines: List[str] = []
    for idx in range(table_start, len(lines)):
        if lines[idx].strip().startswith("|"):
            table_lines.append(lines[idx])
        else:
            break
    return _parse_markdown_table(table_lines)


def _row_by_model(df: pd.DataFrame, model_name: str) -> pd.Series:
    if "model" not in df.columns:
        raise ValueError("Table missing 'model' column")
    mask = df["model"].map(_norm) == _norm(model_name)
    out = df[mask]
    if out.empty:
        raise ValueError(f"Model not found in table: {model_name}")
    return out.iloc[0]


def _row_by_model_split(df: pd.DataFrame, model_name: str, split: str) -> pd.Series:
    if "model" not in df.columns or "split" not in df.columns:
        raise ValueError("Table missing 'model'/'split' columns")
    mask = (df["model"].map(_norm) == _norm(model_name)) & (
        df["split"].map(_norm) == _norm(split)
    )
    out = df[mask]
    if out.empty:
        raise ValueError(f"Model/split not found in table: {model_name} / {split}")
    return out.iloc[0]


def _metric_value(df: pd.DataFrame, metric_name: str) -> float:
    if "metric" not in df.columns or "value" not in df.columns:
        raise ValueError("Table missing 'metric'/'value' columns")
    mask = df["metric"].map(_norm) == _norm(metric_name)
    out = df[mask]
    if out.empty:
        raise ValueError(f"Metric not found in table: {metric_name}")
    return _as_float(out.iloc[0]["value"])


def build_cross_dataset_tables(input_index: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    rows: List[Dict[str, object]] = []

    hyper_text = _read_text(input_index["in_ch3_hyperkvasir_report"])
    hyper_overall = _table_after_heading(hyper_text, "## 3) Overall Test Metrics (all saved model variants)")
    hyper_best = _row_by_model(hyper_overall, "resnet50_supervised")
    hyper_base = _row_by_model(hyper_overall, "blip2_zero_shot_clip")
    rows.append(
        {
            "dataset": "HyperKvasir",
            "score_metric": "accuracy",
            "constrained_model": "resnet50_supervised",
            "constrained_score": _as_float(hyper_best["accuracy"]),
            "baseline_model": "blip2_zero_shot_clip",
            "baseline_score": _as_float(hyper_base["accuracy"]),
            "source_report_path": str(input_index["in_ch3_hyperkvasir_report"]),
        }
    )

    imageclef_text = _read_text(input_index["in_ch3_imageclef_report"])
    imageclef_overall = _table_after_heading(
        imageclef_text,
        "## 3) Overall Metrics (label-id evaluation semantics used by local pipeline)",
    )
    imageclef_best = _row_by_model_split(imageclef_overall, "vilt_finetune", "validation")
    imageclef_base = _row_by_model_split(
        imageclef_overall,
        "qwen2_5_vl_zeroshot_projected",
        "validation",
    )
    rows.append(
        {
            "dataset": "ImageCLEF MEDVQA-GI 2023",
            "score_metric": "accuracy",
            "constrained_model": "vilt_finetune",
            "constrained_score": _as_float(imageclef_best["accuracy"]),
            "baseline_model": "qwen2_5_vl_zeroshot_projected",
            "baseline_score": _as_float(imageclef_base["accuracy"]),
            "source_report_path": str(input_index["in_ch3_imageclef_report"]),
        }
    )

    kvasir_vqa_text = _read_text(input_index["in_ch3_kvasir_vqa_report"])
    kvasir_vqa_yesno = _table_after_heading(
        kvasir_vqa_text,
        "## 4) Yes/No Benchmarks Across Available Runs",
    )
    kvasir_vqa_best = _row_by_model(kvasir_vqa_yesno, "resnet_gru_m1_yesno")
    kvasir_vqa_base = _row_by_model(kvasir_vqa_yesno, "blip2_zeroshot_yesno")
    rows.append(
        {
            "dataset": "Kvasir-VQA (yes/no subset)",
            "score_metric": "accuracy",
            "constrained_model": "resnet_gru_m1_yesno",
            "constrained_score": _as_float(kvasir_vqa_best["accuracy"]),
            "baseline_model": "blip2_zeroshot_yesno",
            "baseline_score": _as_float(kvasir_vqa_base["accuracy"]),
            "source_report_path": str(input_index["in_ch3_kvasir_vqa_report"]),
        }
    )

    kvasir_x1_text = _read_text(input_index["in_ch3_kvasir_vqa_x1_report"])
    kvasir_x1_gen = _table_after_heading(
        kvasir_x1_text,
        "## 3) Generative VQA Leaderboard (Persisted Artifacts)",
    )
    kvasir_x1_best = _row_by_model(kvasir_x1_gen, "medgemma_lora_original")
    kvasir_x1_base = _row_by_model(kvasir_x1_gen, "qwen2_5_vl_zeroshot")
    rows.append(
        {
            "dataset": "Kvasir-VQA-x1 (generative)",
            "score_metric": "token_f1",
            "constrained_model": "medgemma_lora_original",
            "constrained_score": _as_float(kvasir_x1_best["token_f1"]),
            "baseline_model": "qwen2_5_vl_zeroshot",
            "baseline_score": _as_float(kvasir_x1_base["token_f1"]),
            "source_report_path": str(input_index["in_ch3_kvasir_vqa_x1_report"]),
        }
    )

    limuc_text = _read_text(input_index["in_ch3_limuc_report"])
    limuc_overall = _table_after_heading(limuc_text, "## 3) Overall Test Metrics (All Persisted Models)")
    limuc_best = _row_by_model(limuc_overall, "finetune_resnet50")
    limuc_base = _row_by_model(limuc_overall, "vlm_zero_shot_mayo")
    rows.append(
        {
            "dataset": "LIMUC severity",
            "score_metric": "accuracy",
            "constrained_model": "finetune_resnet50",
            "constrained_score": _as_float(limuc_best["accuracy"]),
            "baseline_model": "vlm_zero_shot_mayo",
            "baseline_score": _as_float(limuc_base["accuracy"]),
            "source_report_path": str(input_index["in_ch3_limuc_report"]),
        }
    )

    benchmark_df = pd.DataFrame(rows)
    benchmark_df["absolute_gap"] = (
        benchmark_df["constrained_score"].astype(float)
        - benchmark_df["baseline_score"].astype(float)
    )
    benchmark_df["residual_error_reduction"] = (
        benchmark_df["absolute_gap"] / (1.0 - benchmark_df["baseline_score"].astype(float))
    )
    benchmark_df["dataset_display"] = benchmark_df.apply(
        lambda r: f"{r['dataset']} ({r['score_metric']})", axis=1
    )
    benchmark_df.to_csv(DATA_DIR / "ch3_cross_dataset_benchmark.csv", index=False)

    seg_text = _read_text(input_index["in_ch3_kvasir_seg_report"])
    seg_fg = _table_after_heading(seg_text, "### 1.2 Mask Foreground Coverage Summary")
    seg_comp = _table_after_heading(seg_text, "### 1.3 Mask Connected-component Summary")
    seg_bbox = _table_after_heading(seg_text, "### 1.4 Bounding-box Summary")
    seg_df = pd.DataFrame(
        [
            {
                "dataset": "Kvasir-SEG (supporting)",
                "foreground_ratio_mean": _metric_value(seg_fg, "foreground_ratio_mean"),
                "single_component_share": _metric_value(seg_comp, "single_component_share"),
                "bbox_vs_mask_bbox_iou_mean": _metric_value(seg_bbox, "bbox_vs_mask_bbox_iou_mean"),
                "note": "Supporting segmentation context; no tuned-vs-zero-shot pair in report.",
                "source_report_path": str(input_index["in_ch3_kvasir_seg_report"]),
            }
        ]
    )
    seg_df.to_csv(DATA_DIR / "ch3_kvasir_seg_support_metrics.csv", index=False)

    heatmap_df = benchmark_df[
        [
            "dataset_display",
            "constrained_score",
            "baseline_score",
            "absolute_gap",
            "residual_error_reduction",
        ]
    ].copy()
    heatmap_df.to_csv(DATA_DIR / "ch3_cross_dataset_heatmap_matrix.csv", index=False)

    return {
        "benchmark": benchmark_df,
        "seg_support": seg_df,
        "heatmap": heatmap_df,
    }


def _draw_value_labels(ax: plt.Axes, values: np.ndarray, fontsize: int = 9) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text = "NA" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color="black")


def plot_cross_dataset_heatmap(
    benchmark_df: pd.DataFrame, seg_df: pd.DataFrame
) -> Path:
    left_vals = benchmark_df[
        ["constrained_score", "baseline_score", "absolute_gap", "residual_error_reduction"]
    ].to_numpy(dtype=float)
    left_rows = benchmark_df["dataset_display"].tolist()
    left_cols = [
        "Constrained/\nTuned score",
        "Zero-shot/\nOpen baseline",
        "Absolute gap",
        "Residual-error\nreduction",
    ]

    right_vals = seg_df[
        ["foreground_ratio_mean", "single_component_share", "bbox_vs_mask_bbox_iou_mean"]
    ].to_numpy(dtype=float)
    right_rows = seg_df["dataset"].tolist()
    right_cols = ["FG ratio\nmean", "Single-comp\nshare", "BBox-mask\nIoU mean"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [3.5, 1.5]},
    )

    im_left = axes[0].imshow(left_vals, cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(left_cols)))
    axes[0].set_yticks(np.arange(len(left_rows)))
    axes[0].set_xticklabels(left_cols)
    axes[0].set_yticklabels(left_rows)
    axes[0].set_title("Comparative reliability benchmarks")
    _draw_value_labels(axes[0], left_vals)
    cbar_left = fig.colorbar(im_left, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_left.set_label("Score")

    im_right = axes[1].imshow(right_vals, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(right_cols)))
    axes[1].set_yticks(np.arange(len(right_rows)))
    axes[1].set_xticklabels(right_cols)
    axes[1].set_yticklabels(right_rows)
    axes[1].set_title("Supporting segmentation context")
    _draw_value_labels(axes[1], right_vals)
    cbar_right = fig.colorbar(im_right, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_right.set_label("Score")

    fig.suptitle("Chapter 3 Cross-Dataset Benchmark Heatmap", fontsize=14)
    fig.text(
        0.02,
        0.02,
        "Primary panel compares best constrained/tuned results against zero-shot/open baselines "
        "within each dataset's own metric convention.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = FIG_DIR / "F01_ch3_cross_dataset_benchmark_heatmap.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_part4_summary(outputs: Dict[str, Path]) -> None:
    lines = [
        "# Part 4 (Chapter 3 Figure) Summary",
        "",
        f"- Generated UTC: `{_utc_now()}`",
        "",
        "## Data Outputs",
    ]
    for key in [
        "ch3_cross_dataset_benchmark_csv",
        "ch3_cross_dataset_heatmap_matrix_csv",
        "ch3_kvasir_seg_support_metrics_csv",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.extend(["", "## Figure Outputs"])
    lines.append(f"- `f01_cross_dataset_heatmap_png`: `{outputs['f01_cross_dataset_heatmap_png']}`")
    PART4_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    input_index = _load_freeze_index()
    tables = build_cross_dataset_tables(input_index)
    f01_png = plot_cross_dataset_heatmap(tables["benchmark"], tables["seg_support"])

    outputs = {
        "ch3_cross_dataset_benchmark_csv": DATA_DIR / "ch3_cross_dataset_benchmark.csv",
        "ch3_cross_dataset_heatmap_matrix_csv": DATA_DIR / "ch3_cross_dataset_heatmap_matrix.csv",
        "ch3_kvasir_seg_support_metrics_csv": DATA_DIR / "ch3_kvasir_seg_support_metrics.csv",
        "f01_cross_dataset_heatmap_png": f01_png,
    }
    write_part4_summary(outputs)

    print("[part4] Chapter 3 figure pipeline completed")
    for key, path in outputs.items():
        print(f"[part4] {key}: {path}")
    print(f"[part4] summary: {PART4_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
