#!/usr/bin/env python3
"""Pass 2 analytics for Chapter 4: calibration, significance, ordinal errors, operating points."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from _results_utils import collect_run_records, find_limuc_root, normalize_prediction_df


N_CLASSES = 4


@dataclass
class RunBundle:
    run_name: str
    track: str
    lane: str
    run_dir: Path
    record: Dict[str, Any]
    pred_test: pd.DataFrame
    pred_val: pd.DataFrame | None
    probs_test: np.ndarray | None
    probs_val: np.ndarray | None


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _bh_fdr(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    ranked = np.asarray(p_values, dtype=float)[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adj[i] = value
        prev = value
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()


def _is_generative(record: Dict[str, Any]) -> bool:
    track = str(record.get("track", ""))
    run_name = str(record.get("run_name", "")).lower()
    model = str(record.get("model", "")).lower()
    return track == "3_vlm_severity" or "vlm" in run_name or "lora" in run_name or "vlm" in model


def _lane(record: Dict[str, Any]) -> str:
    return "generative" if _is_generative(record) else "supervised"


def _majority_split_hash(records: Iterable[Dict[str, Any]]) -> str:
    hashes = [str(r.get("split_hash")) for r in records if r.get("split_hash")]
    if not hashes:
        return ""
    vc = pd.Series(hashes).value_counts()
    return str(vc.index[0])


def _prediction_df(run_dir: Path, split: str) -> pd.DataFrame | None:
    path = run_dir / f"pred_{split}.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    norm = normalize_prediction_df(raw)
    if norm is None or norm.empty:
        return None
    norm = norm[norm["y_true"].notna() & norm["y_pred"].notna()].copy()
    norm["y_true"] = pd.to_numeric(norm["y_true"], errors="coerce")
    norm["y_pred"] = pd.to_numeric(norm["y_pred"], errors="coerce")
    norm = norm[norm["y_true"].between(0, 3, inclusive="both")].copy()
    # Match historical Chapter 4 metric behavior: invalid parses map to class 0,
    # while parse-rate remains tracked separately in run artifacts.
    invalid_pred = ~norm["y_pred"].between(0, 3, inclusive="both")
    if invalid_pred.any():
        norm.loc[invalid_pred, "y_pred"] = 0
    norm["y_true"] = norm["y_true"].astype(int)
    norm["y_pred"] = norm["y_pred"].astype(int)
    return norm


def _probability_matrix(df: pd.DataFrame | None) -> np.ndarray | None:
    if df is None or df.empty:
        return None
    candidates = [
        ["prob_0", "prob_1", "prob_2", "prob_3"],
        ["prob_mayo_0", "prob_mayo_1", "prob_mayo_2", "prob_mayo_3"],
        ["p0", "p1", "p2", "p3"],
    ]
    use_cols: List[str] | None = None
    for cols in candidates:
        if all(c in df.columns for c in cols):
            use_cols = cols
            break
    if use_cols is None:
        return None
    probs = df.loc[:, use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(probs).all(axis=1)
    if finite.sum() == 0:
        return None
    probs = probs[finite]
    row_sum = probs.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 0
    if valid.sum() == 0:
        return None
    probs = probs[valid] / row_sum[valid]
    return probs


def _align_probs_to_df(df: pd.DataFrame, probs: np.ndarray | None) -> Tuple[pd.DataFrame, np.ndarray] | Tuple[None, None]:
    if probs is None:
        return None, None
    candidates = [
        ["prob_0", "prob_1", "prob_2", "prob_3"],
        ["prob_mayo_0", "prob_mayo_1", "prob_mayo_2", "prob_mayo_3"],
        ["p0", "p1", "p2", "p3"],
    ]
    use_cols: List[str] | None = None
    for cols in candidates:
        if all(c in df.columns for c in cols):
            use_cols = cols
            break
    if use_cols is None:
        return None, None
    values = df.loc[:, use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    row_sum = np.where(finite, values.sum(axis=1), 0.0)
    valid = finite & (row_sum > 0.0)
    if valid.sum() == 0:
        return None, None
    aligned_df = df.loc[valid].copy()
    aligned_probs = values[valid] / row_sum[valid, None]
    return aligned_df, aligned_probs


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    flat = np.bincount((y_true.astype(int) * n_classes + y_pred.astype(int)), minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def _metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    n = int(cm.sum())
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "balanced_acc": 0.0, "qwk": 0.0}
    tp = np.diag(cm).astype(float)
    row_sum = cm.sum(axis=1).astype(float)
    col_sum = cm.sum(axis=0).astype(float)
    acc = float(tp.sum() / n)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    macro_f1 = float(np.mean(f1))
    balanced_acc = float(np.mean(recall))
    ii, jj = np.indices(cm.shape)
    w = ((ii - jj) ** 2) / float((N_CLASSES - 1) ** 2)
    expected = np.outer(row_sum, col_sum) / max(float(n), 1.0)
    denom = float((w * expected).sum())
    if denom <= 0:
        qwk = 0.0
    else:
        qwk = float(1.0 - ((w * cm).sum() / denom))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "qwk": qwk,
    }


def _metrics_for_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = _confusion(y_true, y_pred)
    metrics = _metrics_from_confusion(cm)
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    return metrics


def _multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _multiclass_nll(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true.astype(int)], eps, 1.0)
    return float(-np.mean(np.log(p)))


def _ece_toplabel(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Tuple[float, pd.DataFrame]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true.astype(int)).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.minimum(np.searchsorted(edges, conf, side="right") - 1, n_bins - 1)
    rows = []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        if c == 0:
            rows.append(
                {
                    "bin_id": b,
                    "bin_left": float(edges[b]),
                    "bin_right": float(edges[b + 1]),
                    "count": 0,
                    "mean_confidence": np.nan,
                    "accuracy": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(acc - mean_conf)
        ece += gap * (c / max(n, 1))
        rows.append(
            {
                "bin_id": b,
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "count": c,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "abs_gap": float(gap),
            }
        )
    return float(ece), pd.DataFrame(rows)


def _apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    z = np.log(np.clip(probs, eps, 1.0))
    z = z / max(float(temperature), eps)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _fit_temperature_grid(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    grid: np.ndarray,
) -> Tuple[float, float]:
    best_t = 1.0
    best_nll = float("inf")
    for t in grid:
        p = _apply_temperature(probs_val, float(t))
        nll = _multiclass_nll(y_val, p)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, best_nll


def _mcnemar_stats(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, float]:
    merged = df_a[["img_id_canonical", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_a"}).merge(
        df_b[["img_id_canonical", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_b"}),
        on=["img_id_canonical", "y_true"],
        how="inner",
    )
    if merged.empty:
        return {
            "n": 0,
            "n01_a_wrong_b_right": 0,
            "n10_a_right_b_wrong": 0,
            "chi2_cc": 0.0,
            "p_value_cc": 1.0,
            "p_value_exact": 1.0,
        }
    y_true = merged["y_true"].astype(int).to_numpy()
    y_a = merged["y_pred_a"].astype(int).to_numpy()
    y_b = merged["y_pred_b"].astype(int).to_numpy()
    a_correct = y_a == y_true
    b_correct = y_b == y_true
    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    if (n01 + n10) > 0:
        chi2_cc = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
        p_cc = math.erfc(math.sqrt(max(chi2_cc, 0.0) / 2.0))
        p_exact = float(binomtest(min(n01, n10), n=n01 + n10, p=0.5, alternative="two-sided").pvalue)
    else:
        chi2_cc = 0.0
        p_cc = 1.0
        p_exact = 1.0
    return {
        "n": int(len(y_true)),
        "n01_a_wrong_b_right": n01,
        "n10_a_right_b_wrong": n10,
        "chi2_cc": float(chi2_cc),
        "p_value_cc": float(p_cc),
        "p_value_exact": float(p_exact),
    }


def _paired_bootstrap_deltas(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    *,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {}
    obs_a = _metrics_for_arrays(y_true, y_a)
    obs_b = _metrics_for_arrays(y_true, y_b)
    metric_names = ["accuracy", "macro_f1", "balanced_acc", "qwk", "mae"]
    obs_delta = {m: obs_a[m] - obs_b[m] for m in metric_names}

    deltas: Dict[str, List[float]] = {m: [] for m in metric_names}
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ya = y_a[idx]
        yb = y_b[idx]
        ma = _metrics_for_arrays(yt, ya)
        mb = _metrics_for_arrays(yt, yb)
        for m in metric_names:
            deltas[m].append(float(ma[m] - mb[m]))

    out: Dict[str, float] = {"n_paired": int(n)}
    for m in metric_names:
        vals = np.asarray(deltas[m], dtype=float)
        out[f"{m}_delta_observed"] = float(obs_delta[m])
        out[f"{m}_delta_mean"] = float(vals.mean())
        out[f"{m}_delta_ci_low"] = float(np.percentile(vals, 2.5))
        out[f"{m}_delta_ci_high"] = float(np.percentile(vals, 97.5))
    return out


def _operating_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    n = int(len(y_true_bin))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    acc = _safe_div(tp + tn, n)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    bal = (recall + specificity) / 2.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "sensitivity_recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal,
        "youden_j": (recall + specificity - 1.0),
    }


def _plot_reliability(
    bins_df: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    work = bins_df[bins_df["count"] > 0].copy()
    if work.empty:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax.plot(work["mean_confidence"], work["accuracy"], marker="o", linewidth=1.5)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_remission_curve(sweep_df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(sweep_df["threshold"], sweep_df["sensitivity_recall"], label="Sensitivity")
    ax.plot(sweep_df["threshold"], sweep_df["specificity"], label="Specificity")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    ax.set_xlabel("Threshold on P(non-remission = Mayo 2/3)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to LIMUC root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <LIMUC>/4_reporting/out",
    )
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=1686,
        help="Expected test rows for full runs.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=15,
        help="Number of bins for ECE/reliability.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=400,
        help="Bootstrap iterations for paired delta CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--remission-threshold-grid",
        type=int,
        default=501,
        help="Number of threshold points in [0,1] for remission operating-point sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    out_dir = args.out_dir.resolve() if args.out_dir else (dataset_root / "4_reporting" / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = collect_run_records(dataset_root=dataset_root, expected_test_rows=args.expected_test_rows)
    full_records = [r for r in records if int(r.get("test_rows") or -1) == int(args.expected_test_rows)]
    if not full_records:
        raise RuntimeError("No full runs found for Pass 2 analytics.")
    canonical_hash = _majority_split_hash(full_records)
    if canonical_hash:
        full_records = [r for r in full_records if str(r.get("split_hash") or "") == canonical_hash]

    bundles: List[RunBundle] = []
    for r in full_records:
        run_name = str(r.get("run_name"))
        run_dir = dataset_root / str(r.get("run_dir"))
        pred_test = _prediction_df(run_dir, "test")
        if pred_test is None or pred_test.empty:
            continue
        pred_val = _prediction_df(run_dir, "val")
        probs_test = _probability_matrix(pred_test)
        probs_val = _probability_matrix(pred_val) if pred_val is not None else None
        bundles.append(
            RunBundle(
                run_name=run_name,
                track=str(r.get("track", "")),
                lane=_lane(r),
                run_dir=run_dir,
                record=r,
                pred_test=pred_test,
                pred_val=pred_val,
                probs_test=probs_test,
                probs_val=probs_val,
            )
        )
    if not bundles:
        raise RuntimeError("Could not load normalized pred_test for any full run.")

    # -------------------------
    # Calibration + temperature scaling
    # -------------------------
    cal_rows: List[Dict[str, Any]] = []
    temp_rows: List[Dict[str, Any]] = []
    bins_rows: List[Dict[str, Any]] = []
    scaled_test_probs: Dict[str, np.ndarray] = {}

    temp_grid = np.exp(np.linspace(np.log(0.05), np.log(5.0), 300))

    for b in bundles:
        test_df, test_probs = _align_probs_to_df(b.pred_test, b.probs_test)
        if test_df is None or test_probs is None:
            continue
        y_test = test_df["y_true"].astype(int).to_numpy()
        pred_prob = test_probs.argmax(axis=1).astype(int)
        raw_metrics = _metrics_for_arrays(y_test, pred_prob)
        raw_ece, raw_bins = _ece_toplabel(y_test, test_probs, n_bins=args.calibration_bins)
        raw_brier = _multiclass_brier(y_test, test_probs)
        raw_nll = _multiclass_nll(y_test, test_probs)
        cal_rows.append(
            {
                "run_name": b.run_name,
                "lane": b.lane,
                "track": b.track,
                "mode": "raw",
                "n": len(y_test),
                "accuracy_argmax": raw_metrics["accuracy"],
                "macro_f1_argmax": raw_metrics["macro_f1"],
                "balanced_acc_argmax": raw_metrics["balanced_acc"],
                "qwk_argmax": raw_metrics["qwk"],
                "ece_toplabel": raw_ece,
                "brier_multiclass": raw_brier,
                "nll_multiclass": raw_nll,
                "temperature": 1.0,
            }
        )
        for row in raw_bins.to_dict(orient="records"):
            bins_rows.append(
                {
                    "run_name": b.run_name,
                    "lane": b.lane,
                    "mode": "raw",
                    **row,
                }
            )

        val_df, val_probs = _align_probs_to_df(b.pred_val, b.probs_val) if b.pred_val is not None else (None, None)
        if val_df is None or val_probs is None:
            continue
        y_val = val_df["y_true"].astype(int).to_numpy()
        best_t, val_nll_raw_or_scaled = _fit_temperature_grid(y_val, val_probs, temp_grid)
        scaled_probs_test = _apply_temperature(test_probs, best_t)
        scaled_test_probs[b.run_name] = scaled_probs_test
        pred_scaled = scaled_probs_test.argmax(axis=1).astype(int)
        scaled_metrics = _metrics_for_arrays(y_test, pred_scaled)
        scaled_ece, scaled_bins = _ece_toplabel(y_test, scaled_probs_test, n_bins=args.calibration_bins)
        scaled_brier = _multiclass_brier(y_test, scaled_probs_test)
        scaled_nll = _multiclass_nll(y_test, scaled_probs_test)

        cal_rows.append(
            {
                "run_name": b.run_name,
                "lane": b.lane,
                "track": b.track,
                "mode": "temperature_scaled",
                "n": len(y_test),
                "accuracy_argmax": scaled_metrics["accuracy"],
                "macro_f1_argmax": scaled_metrics["macro_f1"],
                "balanced_acc_argmax": scaled_metrics["balanced_acc"],
                "qwk_argmax": scaled_metrics["qwk"],
                "ece_toplabel": scaled_ece,
                "brier_multiclass": scaled_brier,
                "nll_multiclass": scaled_nll,
                "temperature": best_t,
            }
        )
        temp_rows.append(
            {
                "run_name": b.run_name,
                "lane": b.lane,
                "n_val": len(y_val),
                "n_test": len(y_test),
                "best_temperature": best_t,
                "val_nll_at_best_temperature": val_nll_raw_or_scaled,
                "test_ece_raw": raw_ece,
                "test_ece_scaled": scaled_ece,
                "test_ece_delta_scaled_minus_raw": scaled_ece - raw_ece,
                "test_nll_raw": raw_nll,
                "test_nll_scaled": scaled_nll,
                "test_nll_delta_scaled_minus_raw": scaled_nll - raw_nll,
                "test_brier_raw": raw_brier,
                "test_brier_scaled": scaled_brier,
                "test_brier_delta_scaled_minus_raw": scaled_brier - raw_brier,
            }
        )
        for row in scaled_bins.to_dict(orient="records"):
            bins_rows.append(
                {
                    "run_name": b.run_name,
                    "lane": b.lane,
                    "mode": "temperature_scaled",
                    **row,
                }
            )

        raw_plot = figures_dir / f"pass2_reliability_raw_{b.run_name}.png"
        scaled_plot = figures_dir / f"pass2_reliability_scaled_{b.run_name}.png"
        _plot_reliability(raw_bins, raw_plot, f"Reliability: {b.run_name} (raw)")
        _plot_reliability(scaled_bins, scaled_plot, f"Reliability: {b.run_name} (temp-scaled)")

    cal_df = pd.DataFrame(cal_rows).sort_values(["ece_toplabel", "run_name", "mode"])
    cal_path = out_dir / "pass2_calibration_summary.csv"
    cal_df.to_csv(cal_path, index=False)

    temp_df = pd.DataFrame(temp_rows).sort_values(["test_ece_delta_scaled_minus_raw", "run_name"])
    temp_path = out_dir / "pass2_temperature_scaling_summary.csv"
    temp_df.to_csv(temp_path, index=False)

    bins_df = pd.DataFrame(bins_rows).sort_values(["run_name", "mode", "bin_id"])
    bins_path = out_dir / "pass2_reliability_bins.csv"
    bins_df.to_csv(bins_path, index=False)

    # -------------------------
    # Ordinal/boundary analysis
    # -------------------------
    ordinal_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    pred_by_run: Dict[str, pd.DataFrame] = {}
    lane_by_run: Dict[str, str] = {}

    for b in bundles:
        df = b.pred_test.copy()
        pred_by_run[b.run_name] = df
        lane_by_run[b.run_name] = b.lane
        y_true = df["y_true"].astype(int).to_numpy()
        y_pred = df["y_pred"].astype(int).to_numpy()
        abs_err = np.abs(y_true - y_pred)
        n = len(df)
        dist_count = {d: int((abs_err == d).sum()) for d in range(4)}
        for d in range(4):
            dist_rows.append(
                {
                    "run_name": b.run_name,
                    "lane": b.lane,
                    "distance": d,
                    "count": dist_count[d],
                    "proportion": _safe_div(dist_count[d], n),
                }
            )
        boundary_01 = int(((y_true == 0) & (y_pred == 1)).sum() + ((y_true == 1) & (y_pred == 0)).sum())
        boundary_12 = int(((y_true == 1) & (y_pred == 2)).sum() + ((y_true == 2) & (y_pred == 1)).sum())
        boundary_23 = int(((y_true == 2) & (y_pred == 3)).sum() + ((y_true == 3) & (y_pred == 2)).sum())
        cross_clinical = int(
            (((y_true <= 1) & (y_pred >= 2)) | ((y_true >= 2) & (y_pred <= 1))).sum()
        )

        ordinal_rows.append(
            {
                "run_name": b.run_name,
                "lane": b.lane,
                "n": n,
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "exact_match_rate": _safe_div(dist_count[0], n),
                "abs_error_ge_1_rate": _safe_div(n - dist_count[0], n),
                "abs_error_ge_2_rate": _safe_div(dist_count[2] + dist_count[3], n),
                "abs_error_eq_3_rate": _safe_div(dist_count[3], n),
                "boundary_0_1_error_rate": _safe_div(boundary_01, n),
                "boundary_1_2_error_rate": _safe_div(boundary_12, n),
                "boundary_2_3_error_rate": _safe_div(boundary_23, n),
                "clinical_cross_boundary_0_1_vs_2_3_rate": _safe_div(cross_clinical, n),
            }
        )
        boundary_rows.append(
            {
                "run_name": b.run_name,
                "lane": b.lane,
                "n": n,
                "true0_pred1": int(((y_true == 0) & (y_pred == 1)).sum()),
                "true1_pred0": int(((y_true == 1) & (y_pred == 0)).sum()),
                "true1_pred2": int(((y_true == 1) & (y_pred == 2)).sum()),
                "true2_pred1": int(((y_true == 2) & (y_pred == 1)).sum()),
                "true2_pred3": int(((y_true == 2) & (y_pred == 3)).sum()),
                "true3_pred2": int(((y_true == 3) & (y_pred == 2)).sum()),
            }
        )

    ordinal_df = pd.DataFrame(ordinal_rows).sort_values(["mae", "run_name"], ascending=[True, True])
    ordinal_path = out_dir / "pass2_ordinal_error_profile.csv"
    ordinal_df.to_csv(ordinal_path, index=False)

    dist_df = pd.DataFrame(dist_rows).sort_values(["run_name", "distance"])
    dist_path = out_dir / "pass2_ordinal_distance_distribution.csv"
    dist_df.to_csv(dist_path, index=False)

    boundary_df = pd.DataFrame(boundary_rows).sort_values("run_name")
    boundary_path = out_dir / "pass2_boundary_confusion_pairs.csv"
    boundary_df.to_csv(boundary_path, index=False)

    # -------------------------
    # Pairwise significance (all full runs)
    # -------------------------
    sig_rows: List[Dict[str, Any]] = []
    for run_a, run_b in combinations(sorted(pred_by_run.keys()), 2):
        stats = _mcnemar_stats(pred_by_run[run_a], pred_by_run[run_b])
        sig_rows.append(
            {
                "pair": f"{run_a} vs {run_b}",
                "run_a": run_a,
                "lane_a": lane_by_run.get(run_a, ""),
                "run_b": run_b,
                "lane_b": lane_by_run.get(run_b, ""),
                **stats,
            }
        )
    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        sig_df["p_value_exact_fdr_bh"] = _bh_fdr(sig_df["p_value_exact"].tolist())
        sig_df["significant_exact_p_lt_0_05"] = (sig_df["p_value_exact"] < 0.05).astype(int)
        sig_df["significant_fdr_bh_lt_0_05"] = (sig_df["p_value_exact_fdr_bh"] < 0.05).astype(int)
        sig_df = sig_df.sort_values(["p_value_exact_fdr_bh", "p_value_exact", "pair"])
    sig_path = out_dir / "pass2_pairwise_mcnemar_all.csv"
    sig_df.to_csv(sig_path, index=False)

    # -------------------------
    # Paired bootstrap deltas (supervised vs generative)
    # -------------------------
    sup_runs = [b.run_name for b in bundles if b.lane == "supervised"]
    gen_runs = [b.run_name for b in bundles if b.lane == "generative"]
    delta_rows: List[Dict[str, Any]] = []
    for sup in sorted(sup_runs):
        for gen in sorted(gen_runs):
            merged = pred_by_run[sup][["img_id_canonical", "y_true", "y_pred"]].rename(
                columns={"y_pred": "y_sup"}
            ).merge(
                pred_by_run[gen][["img_id_canonical", "y_true", "y_pred"]].rename(columns={"y_pred": "y_gen"}),
                on=["img_id_canonical", "y_true"],
                how="inner",
            )
            if merged.empty:
                continue
            y_true = merged["y_true"].astype(int).to_numpy()
            y_sup = merged["y_sup"].astype(int).to_numpy()
            y_gen = merged["y_gen"].astype(int).to_numpy()
            delta = _paired_bootstrap_deltas(
                y_true=y_true,
                y_a=y_sup,
                y_b=y_gen,
                iters=args.bootstrap_iters,
                rng=rng,
            )
            delta_rows.append(
                {
                    "pair": f"{sup} - {gen}",
                    "run_supervised": sup,
                    "run_generative": gen,
                    **delta,
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(["qwk_delta_observed", "accuracy_delta_observed"], ascending=[False, False])
    delta_path = out_dir / "pass2_pairwise_bootstrap_deltas.csv"
    delta_df.to_csv(delta_path, index=False)

    # -------------------------
    # Remission operating points (probability runs)
    # -------------------------
    thresh_values = np.linspace(0.0, 1.0, int(args.remission_threshold_grid))
    sweep_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for b in bundles:
        test_df, test_probs = _align_probs_to_df(b.pred_test, b.probs_test)
        if test_df is None or test_probs is None:
            continue
        y = test_df["y_true"].astype(int).to_numpy()
        y_bin = (y >= 2).astype(int)  # non-remission (active disease) as positive
        for mode, probs in [("raw", test_probs), ("temperature_scaled", scaled_test_probs.get(b.run_name))]:
            if probs is None:
                continue
            score_non_remission = probs[:, 2] + probs[:, 3]
            mode_rows: List[Dict[str, Any]] = []
            for t in thresh_values:
                y_hat = (score_non_remission >= float(t)).astype(int)
                metrics = _operating_metrics(y_bin, y_hat)
                mode_rows.append(
                    {
                        "run_name": b.run_name,
                        "lane": b.lane,
                        "mode": mode,
                        "threshold": float(t),
                        **metrics,
                    }
                )
            mode_df = pd.DataFrame(mode_rows)
            sweep_rows.extend(mode_rows)

            row_05 = mode_df.iloc[(mode_df["threshold"] - 0.5).abs().argmin()]
            row_f1 = mode_df.sort_values(["f1", "youden_j", "threshold"], ascending=[False, False, True]).iloc[0]
            row_youden = mode_df.sort_values(["youden_j", "f1", "threshold"], ascending=[False, False, True]).iloc[0]

            summary_rows.append(
                {
                    "run_name": b.run_name,
                    "lane": b.lane,
                    "mode": mode,
                    "n": int(len(y_bin)),
                    "threshold_at_0_5": float(row_05["threshold"]),
                    "f1_at_0_5": float(row_05["f1"]),
                    "sensitivity_at_0_5": float(row_05["sensitivity_recall"]),
                    "specificity_at_0_5": float(row_05["specificity"]),
                    "best_f1_threshold": float(row_f1["threshold"]),
                    "best_f1": float(row_f1["f1"]),
                    "best_f1_sensitivity": float(row_f1["sensitivity_recall"]),
                    "best_f1_specificity": float(row_f1["specificity"]),
                    "best_youden_threshold": float(row_youden["threshold"]),
                    "best_youden_j": float(row_youden["youden_j"]),
                    "best_youden_sensitivity": float(row_youden["sensitivity_recall"]),
                    "best_youden_specificity": float(row_youden["specificity"]),
                }
            )

            out_curve = figures_dir / f"pass2_remission_operating_curve_{mode}_{b.run_name}.png"
            _plot_remission_curve(mode_df, out_curve, f"Remission Operating Curve: {b.run_name} ({mode})")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = out_dir / "pass2_remission_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)

    op_df = pd.DataFrame(summary_rows).sort_values(["mode", "best_f1", "run_name"], ascending=[True, False, True])
    op_path = out_dir / "pass2_remission_operating_points_summary.csv"
    op_df.to_csv(op_path, index=False)

    # -------------------------
    # Consolidated report
    # -------------------------
    checks = {
        "full_run_predictions_loaded": len(pred_by_run) == len(bundles),
        "pairwise_significance_complete": len(sig_df) == (len(pred_by_run) * (len(pred_by_run) - 1)) // 2,
        "ordinal_profile_complete": len(ordinal_df) == len(bundles),
        "calibration_computed_for_prob_runs": len(cal_df) > 0,
        "temperature_scaling_computed": len(temp_df) > 0,
        "remission_operating_points_computed": len(op_df) > 0,
        "bootstrap_supervised_vs_generative_computed": len(delta_df) > 0,
    }
    status = "PASS" if all(checks.values()) else "WARN"

    best_sup = max((b for b in bundles if b.lane == "supervised"), key=lambda x: float(x.record.get("accuracy") or -1), default=None)
    best_gen = max((b for b in bundles if b.lane == "generative"), key=lambda x: float(x.record.get("macro_f1") or -1), default=None)

    report_payload = {
        "generated_utc": _utc_now(),
        "status": status,
        "dataset_root": str(dataset_root),
        "canonical_split_hash": canonical_hash,
        "n_full_runs_used": len(bundles),
        "n_supervised_runs": len(sup_runs),
        "n_generative_runs": len(gen_runs),
        "checks": checks,
        "outputs": {
            "pass2_calibration_summary_csv": str(cal_path),
            "pass2_temperature_scaling_summary_csv": str(temp_path),
            "pass2_reliability_bins_csv": str(bins_path),
            "pass2_ordinal_error_profile_csv": str(ordinal_path),
            "pass2_ordinal_distance_distribution_csv": str(dist_path),
            "pass2_boundary_confusion_pairs_csv": str(boundary_path),
            "pass2_pairwise_mcnemar_all_csv": str(sig_path),
            "pass2_pairwise_bootstrap_deltas_csv": str(delta_path),
            "pass2_remission_threshold_sweep_csv": str(sweep_path),
            "pass2_remission_operating_points_summary_csv": str(op_path),
        },
        "key_runs": {
            "best_supervised_by_accuracy": best_sup.run_name if best_sup else None,
            "best_generative_by_macro_f1": best_gen.run_name if best_gen else None,
        },
        "summary_counts": {
            "calibration_rows": int(len(cal_df)),
            "temperature_rows": int(len(temp_df)),
            "significance_pairs": int(len(sig_df)),
            "delta_pairs": int(len(delta_df)),
            "operating_point_rows": int(len(op_df)),
        },
    }
    report_json = out_dir / "pass2_chapter4_analytics_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    lines = [
        f"# Pass 2 Chapter 4 Analytics ({status})",
        "",
        f"- generated_utc: `{report_payload['generated_utc']}`",
        f"- canonical_split_hash: `{canonical_hash}`",
        f"- full runs used: `{len(bundles)}`",
        "",
        "## Checklist",
        f"- [{'x' if checks['full_run_predictions_loaded'] else ' '}] full-run predictions loaded",
        f"- [{'x' if checks['pairwise_significance_complete'] else ' '}] pairwise significance all-vs-all",
        f"- [{'x' if checks['ordinal_profile_complete'] else ' '}] ordinal profile for all full runs",
        f"- [{'x' if checks['calibration_computed_for_prob_runs'] else ' '}] calibration metrics computed",
        f"- [{'x' if checks['temperature_scaling_computed'] else ' '}] temperature scaling evaluated",
        f"- [{'x' if checks['remission_operating_points_computed'] else ' '}] remission operating-point sweep",
        f"- [{'x' if checks['bootstrap_supervised_vs_generative_computed'] else ' '}] bootstrap deltas supervised vs generative",
        "",
        "## Key Output Files",
        f"- `{cal_path}`",
        f"- `{temp_path}`",
        f"- `{bins_path}`",
        f"- `{sig_path}`",
        f"- `{delta_path}`",
        f"- `{ordinal_path}`",
        f"- `{boundary_path}`",
        f"- `{sweep_path}`",
        f"- `{op_path}`",
    ]
    report_md = out_dir / "pass2_chapter4_analytics_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {cal_path}")
    print(f"Wrote: {temp_path}")
    print(f"Wrote: {bins_path}")
    print(f"Wrote: {ordinal_path}")
    print(f"Wrote: {dist_path}")
    print(f"Wrote: {boundary_path}")
    print(f"Wrote: {sig_path}")
    print(f"Wrote: {delta_path}")
    print(f"Wrote: {sweep_path}")
    print(f"Wrote: {op_path}")
    print(f"Wrote: {report_json}")
    print(f"Wrote: {report_md}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
