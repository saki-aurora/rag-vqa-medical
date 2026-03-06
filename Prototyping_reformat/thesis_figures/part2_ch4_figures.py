#!/usr/bin/env python3
"""Part 2: Build Chapter 4 figure data and plots."""

from __future__ import annotations

import csv
import json
import math
import re
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


MODEL_SPECS = [
    {
        "run_name": "clip_linear_baseline",
        "display_name": "CLIP linear",
        "family": "frozen",
        "metrics_input_id": "in_ch4_metrics_clip_linear",
        "pred_input_id": "in_ch4_pred_clip_linear",
    },
    {
        "run_name": "resnet50_frozen_logreg",
        "display_name": "ResNet50 frozen+LR",
        "family": "frozen",
        "metrics_input_id": "in_ch4_metrics_resnet50_frozen",
        "pred_input_id": "in_ch4_pred_resnet50_frozen",
    },
    {
        "run_name": "vit_frozen_logreg",
        "display_name": "ViT frozen+LR",
        "family": "frozen",
        "metrics_input_id": "in_ch4_metrics_vit_frozen",
        "pred_input_id": "in_ch4_pred_vit_frozen",
    },
    {
        "run_name": "finetune_resnet50",
        "display_name": "ResNet50 fine-tune",
        "family": "supervised",
        "metrics_input_id": "in_ch4_metrics_finetune_resnet50",
        "pred_input_id": "in_ch4_pred_finetune_resnet50",
    },
    {
        "run_name": "finetune_vit_or_swin",
        "display_name": "ViT/Swin fine-tune",
        "family": "supervised",
        "metrics_input_id": "in_ch4_metrics_finetune_vit_or_swin",
        "pred_input_id": "in_ch4_pred_finetune_vit_or_swin",
    },
    {
        "run_name": "vlm_zero_shot_mayo",
        "display_name": "VLM zero-shot",
        "family": "generative",
        "metrics_input_id": "in_ch4_metrics_vlm_zero_shot",
        "pred_input_id": "in_ch4_pred_vlm_zero_shot",
    },
    {
        "run_name": "vlm_zero_shot_mode2_label_sampling_full_20260302",
        "display_name": "VLM mode2 sampling",
        "family": "generative",
        "metrics_input_id": "in_ch4_metrics_vlm_zero_shot_mode2_sampling",
        "pred_input_id": "in_ch4_pred_vlm_zero_shot_mode2_sampling",
    },
    {
        "run_name": "vlm_lora_finetune_mayo_balanced_full_20260303",
        "display_name": "VLM LoRA balanced",
        "family": "generative",
        "metrics_input_id": "in_ch4_metrics_vlm_lora_balanced_full",
        "pred_input_id": "in_ch4_pred_vlm_lora_balanced_full",
    },
]


RADAR_RUNS = [
    "finetune_resnet50",
    "vit_frozen_logreg",
    "vlm_zero_shot_mode2_label_sampling_full_20260302",
    "vlm_lora_finetune_mayo_balanced_full_20260303",
]


METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced Acc",
    "qwk": "QWK",
    "mae": "MAE",
    "rmse": "RMSE",
}


COLOR_BY_FAMILY = {
    "frozen": "#5B8FF9",
    "supervised": "#61DDAA",
    "generative": "#F6BD16",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _canonical_image_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    if "/" in text or "\\" in text:
        text = Path(text).stem

    text = text.lower()
    text = re.sub(r"^mayo_[0-3]__", "", text)
    text = re.sub(r"_\d{4,}$", "", text)
    return text


def _normalize_prediction_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map: Dict[str, str] = {}
    if "y_true" not in df.columns and "true_label" in df.columns:
        rename_map["true_label"] = "y_true"
    if "y_pred" not in df.columns and "pred_label" in df.columns:
        rename_map["pred_label"] = "y_pred"
    if "img_id" not in df.columns and "image_id" in df.columns:
        rename_map["image_id"] = "img_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Could not normalize y_true/y_pred columns for: {path}")

    key_col = None
    for candidate in ["img_id", "image_id", "image_path"]:
        if candidate in df.columns:
            key_col = candidate
            break
    if key_col is None:
        df = df.reset_index().rename(columns={"index": "row_id"})
        key_col = "row_id"

    out = pd.DataFrame()
    out["key_base"] = df[key_col].map(_canonical_image_id)
    # Preserve duplicate base IDs deterministically using occurrence index.
    out["key_occ"] = out.groupby("key_base").cumcount().astype(str)
    out["key"] = out["key_base"] + "__" + out["key_occ"]
    out["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    out = out.dropna(subset=["key", "y_true", "y_pred"]).copy()
    out["y_true"] = out["y_true"].astype(int)
    out["y_pred"] = out["y_pred"].astype(int)
    out = out[out["key_base"] != ""]
    return out[["key", "y_true", "y_pred"]].reset_index(drop=True)


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


def build_core_metrics(input_index: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for spec in MODEL_SPECS:
        path = input_index[spec["metrics_input_id"]]
        obj = _read_json(path)
        summary = obj.get("summary", obj)
        rows.append(
            {
                "run_name": spec["run_name"],
                "display_name": spec["display_name"],
                "family": spec["family"],
                "metrics_path": str(path),
                "accuracy": summary.get("accuracy"),
                "macro_f1": summary.get("macro_f1"),
                "balanced_accuracy": summary.get("balanced_accuracy", summary.get("balanced_acc")),
                "qwk": summary.get("qwk"),
                "mae": summary.get("mae"),
                "rmse": summary.get("rmse"),
                "parse_rate": summary.get("parse_rate"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "ch4_core_metrics.csv", index=False)
    return df


def plot_core_metrics(df: pd.DataFrame) -> Path:
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "qwk", "mae", "rmse"]
    order = list(df["display_name"])
    y_pos = np.arange(len(order))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = df[metric].astype(float).to_numpy()
        colors = [COLOR_BY_FAMILY[f] for f in df["family"]]
        ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.3)
        if metric in ["mae", "rmse"]:
            ax.invert_xaxis()
            ax.set_xlabel("Lower is better")
        else:
            ax.set_xlabel("Higher is better")
        ax.set_yticks(y_pos)
        if idx % 3 == 0:
            ax.set_yticklabels(order)
        else:
            ax.set_yticklabels([])
        ax.set_title(METRIC_DISPLAY[metric])
        ax.grid(axis="x", alpha=0.25)

    legend_handles = []
    legend_labels = []
    for family in ["frozen", "supervised", "generative"]:
        legend_handles.append(
            plt.Line2D([0], [0], color=COLOR_BY_FAMILY[family], lw=8)
        )
        legend_labels.append(family)
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3)
    fig.suptitle("Chapter 4 Core Metrics Across Model Families", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = FIG_DIR / "F02_ch4_core_metric_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def build_radar_data(df: pd.DataFrame) -> pd.DataFrame:
    radar_metrics = ["accuracy", "macro_f1", "balanced_accuracy", "qwk", "mae", "rmse"]
    lower_is_better = {"mae", "rmse"}

    sel = df[df["run_name"].isin(RADAR_RUNS)].copy()
    rows: List[Dict] = []

    for metric in radar_metrics:
        vals = sel[metric].astype(float).to_numpy()
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        denom = mx - mn
        for _, rec in sel.iterrows():
            raw = float(rec[metric])
            if denom == 0:
                norm = 1.0
            elif metric in lower_is_better:
                norm = (mx - raw) / denom
            else:
                norm = (raw - mn) / denom
            rows.append(
                {
                    "run_name": rec["run_name"],
                    "display_name": rec["display_name"],
                    "family": rec["family"],
                    "metric": metric,
                    "metric_display": METRIC_DISPLAY[metric],
                    "raw_value": raw,
                    "normalized_value": float(norm),
                }
            )

    radar_df = pd.DataFrame(rows)
    radar_df.to_csv(DATA_DIR / "ch4_radar_normalized.csv", index=False)
    return radar_df


def plot_radar(radar_df: pd.DataFrame) -> Path:
    metric_order = ["accuracy", "macro_f1", "balanced_accuracy", "qwk", "mae", "rmse"]
    labels = [METRIC_DISPLAY[m] for m in metric_order]
    theta = np.linspace(0, 2 * np.pi, len(metric_order), endpoint=False)
    theta = np.append(theta, theta[0])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)

    for run_name in RADAR_RUNS:
        sub = radar_df[radar_df["run_name"] == run_name]
        sub = sub.set_index("metric").loc[metric_order].reset_index()
        vals = sub["normalized_value"].to_numpy()
        vals = np.append(vals, vals[0])
        display_name = sub["display_name"].iloc[0]
        family = sub["family"].iloc[0]
        color = COLOR_BY_FAMILY[family]
        ax.plot(theta, vals, linewidth=2, label=display_name, color=color)
        ax.fill(theta, vals, alpha=0.08, color=color)

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Chapter 4 Radar Profile (Normalized)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()

    out_path = FIG_DIR / "F03_ch4_radar_profile.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_remission_slice(input_index: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for spec in MODEL_SPECS:
        path = input_index[spec["pred_input_id"]]
        df = _normalize_prediction_df(path)
        y_true = (df["y_true"] <= 1).astype(int)
        y_pred = (df["y_pred"] <= 1).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        n = int(len(df))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        acc = _safe_div(tp + tn, n)

        rows.append(
            {
                "run_name": spec["run_name"],
                "display_name": spec["display_name"],
                "family": spec["family"],
                "n": n,
                "remission_accuracy": acc,
                "remission_precision": precision,
                "remission_recall_sensitivity": recall,
                "remission_specificity": specificity,
                "remission_f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    out_df = pd.DataFrame(rows).sort_values("remission_accuracy", ascending=False)
    out_df.to_csv(DATA_DIR / "ch4_remission_slice.csv", index=False)
    return out_df


def plot_remission_slice(df: pd.DataFrame) -> Path:
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.bar(
        x - width,
        df["remission_recall_sensitivity"],
        width=width,
        label="Sensitivity",
        color="#4E79A7",
    )
    ax.bar(
        x,
        df["remission_specificity"],
        width=width,
        label="Specificity",
        color="#F28E2B",
    )
    ax.bar(
        x + width,
        df["remission_f1"],
        width=width,
        label="F1",
        color="#59A14F",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Chapter 4 Remission Slice Metrics (Mayo 0-1 vs 2-3)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = FIG_DIR / "F04_ch4_remission_slice_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _mcnemar_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, float]:
    merged = df_a.merge(df_b, on="key", suffixes=("_a", "_b"))
    if merged.empty:
        return {
            "n": 0,
            "n01_a_wrong_b_right": 0,
            "n10_a_right_b_wrong": 0,
            "chi2_cc": 0.0,
            "p_value": 1.0,
        }

    merged = merged[merged["y_true_a"] == merged["y_true_b"]].copy()
    if merged.empty:
        return {
            "n": 0,
            "n01_a_wrong_b_right": 0,
            "n10_a_right_b_wrong": 0,
            "chi2_cc": 0.0,
            "p_value": 1.0,
        }

    a_correct = merged["y_true_a"] == merged["y_pred_a"]
    b_correct = merged["y_true_b"] == merged["y_pred_b"]
    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    disc = n01 + n10
    if disc == 0:
        chi2_cc = 0.0
        p_value = 1.0
    else:
        chi2_cc = (abs(n01 - n10) - 1) ** 2 / disc
        p_value = math.erfc(math.sqrt(chi2_cc / 2.0))

    return {
        "n": int(len(merged)),
        "n01_a_wrong_b_right": n01,
        "n10_a_right_b_wrong": n10,
        "chi2_cc": float(chi2_cc),
        "p_value": float(p_value),
    }


def build_mcnemar(input_index: Dict[str, Path]) -> Dict[str, Path]:
    pred_dfs: Dict[str, pd.DataFrame] = {}
    display_name: Dict[str, str] = {}
    for spec in MODEL_SPECS:
        pred_dfs[spec["run_name"]] = _normalize_prediction_df(
            input_index[spec["pred_input_id"]]
        )
        display_name[spec["run_name"]] = spec["display_name"]

    pairs: List[Dict] = []
    run_names = [spec["run_name"] for spec in MODEL_SPECS]
    for i, run_a in enumerate(run_names):
        for run_b in run_names:
            if run_a == run_b:
                continue
            stats = _mcnemar_pair(pred_dfs[run_a], pred_dfs[run_b])
            pairs.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "display_a": display_name[run_a],
                    "display_b": display_name[run_b],
                    "n": stats["n"],
                    "n01_a_wrong_b_right": stats["n01_a_wrong_b_right"],
                    "n10_a_right_b_wrong": stats["n10_a_right_b_wrong"],
                    "chi2_cc": stats["chi2_cc"],
                    "p_value": stats["p_value"],
                }
            )

    pair_df = pd.DataFrame(pairs)
    pair_path = DATA_DIR / "ch4_mcnemar_pairs.csv"
    pair_df.to_csv(pair_path, index=False)

    matrix = pd.DataFrame(
        np.ones((len(run_names), len(run_names))),
        index=run_names,
        columns=run_names,
        dtype=float,
    )
    for _, row in pair_df.iterrows():
        matrix.loc[row["run_a"], row["run_b"]] = float(row["p_value"])

    matrix.insert(0, "run_name", matrix.index)
    matrix_path = DATA_DIR / "ch4_mcnemar_matrix_wide.csv"
    matrix.to_csv(matrix_path, index=False)

    return {"pairs": pair_path, "matrix": matrix_path}


def plot_mcnemar_heatmap(matrix_path: Path) -> Path:
    mat_df = pd.read_csv(matrix_path)
    run_names = mat_df["run_name"].tolist()
    values = mat_df.drop(columns=["run_name"]).to_numpy(dtype=float)
    values = np.clip(values, 1e-300, 1.0)
    score = -np.log10(values)

    labels = []
    name_to_display = {spec["run_name"]: spec["display_name"] for spec in MODEL_SPECS}
    for run_name in run_names:
        labels.append(name_to_display.get(run_name, run_name))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(score, cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Chapter 4 Pairwise Significance Heatmap (-log10 p)")

    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            if i == j:
                text = "-"
            else:
                p = values[i, j]
                text = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10(p-value)")
    fig.tight_layout()

    out_path = FIG_DIR / "F05_ch4_mcnemar_significance_heatmap.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_confusion_panel(input_index: Dict[str, Path]) -> Path:
    sup_path = input_index["in_ch4_confusion_best_supervised"]
    gen_path = input_index["in_ch4_confusion_best_generative"]

    sup_img = mpimg.imread(sup_path)
    gen_img = mpimg.imread(gen_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(sup_img)
    axes[0].set_title("Best Supervised: ResNet50 fine-tune")
    axes[0].axis("off")

    axes[1].imshow(gen_img)
    axes[1].set_title("Best Generative: VLM LoRA balanced")
    axes[1].axis("off")

    fig.suptitle("Chapter 4 Confusion Matrix Panel")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = FIG_DIR / "F06_ch4_confusion_panel.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_part2_summary(outputs: Dict[str, Path]) -> None:
    lines = [
        "# Part 2 (Chapter 4 Figures) Summary",
        "",
        f"- Generated UTC: `{_utc_now()}`",
        "",
        "## Data Outputs",
    ]
    for key in [
        "ch4_core_metrics_csv",
        "ch4_radar_normalized_csv",
        "ch4_remission_slice_csv",
        "ch4_mcnemar_pairs_csv",
        "ch4_mcnemar_matrix_csv",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.extend(["", "## Figure Outputs"])
    for key in [
        "f02_core_metrics_png",
        "f03_radar_png",
        "f04_remission_png",
        "f05_mcnemar_png",
        "f06_confusion_panel_png",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    PART2_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    input_index = _load_freeze_index()

    core_df = build_core_metrics(input_index)
    core_png = plot_core_metrics(core_df)

    radar_df = build_radar_data(core_df)
    radar_png = plot_radar(radar_df)

    remission_df = build_remission_slice(input_index)
    remission_png = plot_remission_slice(remission_df)

    mcnemar_paths = build_mcnemar(input_index)
    mcnemar_png = plot_mcnemar_heatmap(mcnemar_paths["matrix"])

    confusion_png = plot_confusion_panel(input_index)

    outputs = {
        "ch4_core_metrics_csv": DATA_DIR / "ch4_core_metrics.csv",
        "ch4_radar_normalized_csv": DATA_DIR / "ch4_radar_normalized.csv",
        "ch4_remission_slice_csv": DATA_DIR / "ch4_remission_slice.csv",
        "ch4_mcnemar_pairs_csv": mcnemar_paths["pairs"],
        "ch4_mcnemar_matrix_csv": mcnemar_paths["matrix"],
        "f02_core_metrics_png": core_png,
        "f03_radar_png": radar_png,
        "f04_remission_png": remission_png,
        "f05_mcnemar_png": mcnemar_png,
        "f06_confusion_panel_png": confusion_png,
    }
    write_part2_summary(outputs)

    print("[part2] Chapter 4 figure pipeline completed")
    for key, path in outputs.items():
        print(f"[part2] {key}: {path}")
    print(f"[part2] summary: {PART2_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
