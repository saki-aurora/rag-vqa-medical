#!/usr/bin/env python3
"""Part 2: Build Chapter 4 frozen figure data and plots (Pass5/Pass6/Pass7)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent
OUT_DIR = WORKSPACE_DIR / "out"
DATA_DIR = WORKSPACE_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FREEZE_MANIFEST_PATH = OUT_DIR / "freeze_manifest.json"
PART2_SUMMARY_PATH = OUT_DIR / "part2_ch4_summary.md"


LANE_ORDER = [
    "supervised_pass5",
    "generative_mode1_pass6",
    "generative_mode2_pass6",
]

LANE_DISPLAY = {
    "supervised_pass5": "Supervised\nPass5",
    "generative_mode1_pass6": "Generative mode1\nPass6",
    "generative_mode2_pass6": "Generative mode2\nPass6",
}

LANE_COLOR = {
    "supervised_pass5": "#4E79A7",
    "generative_mode1_pass6": "#59A14F",
    "generative_mode2_pass6": "#E15759",
}

METRIC_ORDER = ["accuracy", "macro_f1", "balanced_accuracy", "qwk"]
METRIC_LABEL = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced Acc",
    "qwk": "QWK",
}

PASS7_LANE_ORDER = ["resnet50_supervised", "vlm_lora_mode1", "vlm_lora_mode2"]
PASS7_LANE_LABEL = {
    "resnet50_supervised": "ResNet50",
    "vlm_lora_mode1": "VLM mode1",
    "vlm_lora_mode2": "VLM mode2",
}
PASS7_LANE_COLOR = {
    "resnet50_supervised": "#4E79A7",
    "vlm_lora_mode1": "#59A14F",
    "vlm_lora_mode2": "#E15759",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _load_internal_metrics(input_index: Dict[str, Path]) -> pd.DataFrame:
    p5 = pd.read_csv(input_index["in_ch4_pass5_metric_summary"])
    p6 = pd.read_csv(input_index["in_ch4_pass6_metric_summary"])

    rows: List[Dict] = []

    p5_sel = p5[p5["metric"].isin(METRIC_ORDER)].copy()
    for _, rec in p5_sel.iterrows():
        rows.append(
            {
                "lane": "supervised_pass5",
                "metric": str(rec["metric"]),
                "mean": float(rec["mean"]),
                "ci95_low": float(rec["ci95_low"]),
                "ci95_high": float(rec["ci95_high"]),
            }
        )

    p6_sel = p6[p6["metric"].isin(METRIC_ORDER)].copy()
    for lane_src, lane_dst in [
        ("lora_mode1_train", "generative_mode1_pass6"),
        ("lora_mode2_eval", "generative_mode2_pass6"),
    ]:
        lane_df = p6_sel[p6_sel["lane"] == lane_src]
        for _, rec in lane_df.iterrows():
            rows.append(
                {
                    "lane": lane_dst,
                    "metric": str(rec["metric"]),
                    "mean": float(rec["mean"]),
                    "ci95_low": float(rec["ci95_low"]),
                    "ci95_high": float(rec["ci95_high"]),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(DATA_DIR / "ch4_frozen_internal_metrics.csv", index=False)
    return out_df


def plot_f02_internal_kpi(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    x = np.arange(len(LANE_ORDER))

    for idx, metric in enumerate(METRIC_ORDER):
        ax = axes[idx]
        sub = df[df["metric"] == metric].set_index("lane")
        vals = [float(sub.loc[lane, "mean"]) for lane in LANE_ORDER]
        colors = [LANE_COLOR[lane] for lane in LANE_ORDER]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([LANE_DISPLAY[l] for l in LANE_ORDER], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(METRIC_LABEL[metric])
        ax.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Chapter 4 Frozen Internal KPI Comparison (Pass5 vs Pass6)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = FIG_DIR / "F02_ch4_core_metric_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_f03_ci_errorbars(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(17, 4.8), sharey=True)

    for idx, metric in enumerate(METRIC_ORDER):
        ax = axes[idx]
        sub = df[df["metric"] == metric].set_index("lane")
        x = np.arange(len(LANE_ORDER))
        means = np.array([float(sub.loc[lane, "mean"]) for lane in LANE_ORDER], dtype=float)
        lows = np.array([float(sub.loc[lane, "ci95_low"]) for lane in LANE_ORDER], dtype=float)
        highs = np.array([float(sub.loc[lane, "ci95_high"]) for lane in LANE_ORDER], dtype=float)
        yerr = np.vstack([means - lows, highs - means])

        ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=4, color="#1f2937", ecolor="#374151", lw=1.2)
        ax.scatter(x, means, s=45, color=[LANE_COLOR[l] for l in LANE_ORDER], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([LANE_DISPLAY[l] for l in LANE_ORDER], fontsize=8, rotation=15, ha="right")
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    fig.suptitle("Chapter 4 Frozen Metric Means with 95% CI", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = FIG_DIR / "F03_ch4_radar_profile.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _load_pass7_drop_subset(input_index: Dict[str, Path]) -> pd.DataFrame:
    drop = pd.read_csv(input_index["in_ch4_pass7_drop_table"])
    keep_metrics = ["accuracy", "macro_f1", "qwk"]
    out = drop[
        drop["lane"].isin(PASS7_LANE_ORDER) & drop["metric"].isin(keep_metrics)
    ].copy()
    out.to_csv(DATA_DIR / "ch4_pass7_drop_subset.csv", index=False)
    return out


def plot_f04_external_drop(drop_df: pd.DataFrame) -> Path:
    metrics = ["accuracy", "macro_f1", "qwk"]
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, lane in enumerate(PASS7_LANE_ORDER):
        sub = drop_df[drop_df["lane"] == lane].set_index("metric")
        vals = np.array([float(sub.loc[m, "delta_external_minus_internal"]) for m in metrics], dtype=float)
        pos = x + (i - 1) * width
        bars = ax.bar(
            pos,
            vals,
            width=width,
            label=PASS7_LANE_LABEL[lane],
            color=PASS7_LANE_COLOR[lane],
            edgecolor="black",
            linewidth=0.3,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v - 0.02 if v < 0 else v + 0.02, f"{v:.3f}", ha="center", va="top" if v < 0 else "bottom", fontsize=8)

    ax.axhline(0.0, color="#111827", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylabel("External - Internal")
    ax.set_title("Chapter 4 External Stress-Test Drops (Pass7 HyperKvasir UC Proxy)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()

    out_path = FIG_DIR / "F04_ch4_remission_slice_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _load_mode1_qc(input_index: Dict[str, Path]) -> pd.DataFrame:
    qc = pd.read_csv(input_index["in_ch4_pass6_mode1_qc"]).copy()
    qc = qc.sort_values("seed").reset_index(drop=True)
    qc.to_csv(DATA_DIR / "ch4_pass6_mode1_qc.csv", index=False)
    return qc


def plot_f05_mode1_qc(qc_df: pd.DataFrame) -> Path:
    x = np.arange(len(qc_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars_acc = ax.bar(x - width / 2.0, qc_df["accuracy"].astype(float), width=width, label="Accuracy", color="#4E79A7")
    bars_qwk = ax.bar(x + width / 2.0, qc_df["qwk"].astype(float), width=width, label="QWK", color="#59A14F")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {int(s)}" for s in qc_df["seed"].tolist()])
    ax.set_ylim(0, 1.0)
    ax.set_title("Chapter 4 Pass6 Mode1 Convergence QC by Seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    for i, (_, row) in enumerate(qc_df.iterrows()):
        pass_tag = "PASS" if bool(row.get("qc_pass", False)) else "FAIL"
        cls = int(row.get("pred_unique_classes", 0))
        ax.text(i, 0.985, f"{pass_tag} | classes={cls}", ha="center", va="top", fontsize=8)

    for bars in [bars_acc, bars_qwk]:
        for b in bars:
            v = float(b.get_height())
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = FIG_DIR / "F05_ch4_mcnemar_significance_heatmap.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_f06_confusion_panel(input_index: Dict[str, Path]) -> Path:
    sup_path = input_index["in_ch4_pass5_confusion_aggregate_png"]
    m1_path = input_index["in_ch4_pass6_confusion_mode1_png"]
    m2_path = input_index["in_ch4_pass6_confusion_mode2_png"]

    sup_img = mpimg.imread(sup_path)
    m1_img = mpimg.imread(m1_path)
    m2_img = mpimg.imread(m2_path)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    axes[0].imshow(sup_img)
    axes[0].set_title("Pass5 supervised\naggregate confusion")
    axes[0].axis("off")

    axes[1].imshow(m1_img)
    axes[1].set_title("Pass6 mode1\naggregate confusion")
    axes[1].axis("off")

    axes[2].imshow(m2_img)
    axes[2].set_title("Pass6 mode2\naggregate confusion")
    axes[2].axis("off")

    fig.suptitle("Chapter 4 Aggregate Confusion Panel (Pass5/Pass6)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = FIG_DIR / "F06_ch4_confusion_panel.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_part2_summary(outputs: Dict[str, Path]) -> None:
    lines = [
        "# Part 2 (Chapter 4 Frozen Figures) Summary",
        "",
        f"- Generated UTC: `{_utc_now()}`",
        "- Input policy: Pass5/Pass6/Pass7 frozen artifacts",
        "",
        "## Data Outputs",
    ]
    for key in [
        "ch4_frozen_internal_metrics_csv",
        "ch4_pass7_drop_subset_csv",
        "ch4_pass6_mode1_qc_csv",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.extend(["", "## Figure Outputs"])
    for key in [
        "f02_core_metrics_png",
        "f03_ci_plot_png",
        "f04_external_drop_png",
        "f05_mode1_qc_png",
        "f06_confusion_panel_png",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    PART2_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    input_index = _load_freeze_index()

    internal_df = _load_internal_metrics(input_index)
    drop_df = _load_pass7_drop_subset(input_index)
    qc_df = _load_mode1_qc(input_index)

    f02 = plot_f02_internal_kpi(internal_df)
    f03 = plot_f03_ci_errorbars(internal_df)
    f04 = plot_f04_external_drop(drop_df)
    f05 = plot_f05_mode1_qc(qc_df)
    f06 = plot_f06_confusion_panel(input_index)

    outputs = {
        "ch4_frozen_internal_metrics_csv": DATA_DIR / "ch4_frozen_internal_metrics.csv",
        "ch4_pass7_drop_subset_csv": DATA_DIR / "ch4_pass7_drop_subset.csv",
        "ch4_pass6_mode1_qc_csv": DATA_DIR / "ch4_pass6_mode1_qc.csv",
        "f02_core_metrics_png": f02,
        "f03_ci_plot_png": f03,
        "f04_external_drop_png": f04,
        "f05_mode1_qc_png": f05,
        "f06_confusion_panel_png": f06,
    }
    write_part2_summary(outputs)

    print("[part2] Chapter 4 frozen figure pipeline completed")
    for key, path in outputs.items():
        print(f"[part2] {key}: {path}")
    print(f"[part2] summary: {PART2_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
