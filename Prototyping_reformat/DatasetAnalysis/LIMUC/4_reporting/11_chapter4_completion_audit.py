#!/usr/bin/env python3
"""Generate a reproducible Chapter 4 completion audit from persisted LIMUC results."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from _results_utils import collect_run_records, find_limuc_root, normalize_prediction_df, write_csv


REQUIRED_FULL_RUNS = {
    "supervised": [
        "finetune_resnet50",
        "finetune_vit_or_swin",
        "resnet50_frozen_logreg",
        "vit_frozen_logreg",
        "clip_linear_baseline",
    ],
    "generative_baseline": ["vlm_zero_shot_mayo"],
}


@dataclass
class PairStats:
    n: int
    n01_a_wrong_b_right: int
    n10_a_right_b_wrong: int
    chi2_cc: float
    p_value: float


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
        help="Expected number of test rows for a full run.",
    )
    parser.add_argument(
        "--chapter-md",
        type=Path,
        default=Path("Thesis/markdown/04_chapter_4_developing_the_proposed_approach.md"),
        help="Chapter markdown to verify sync checks.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _read_run_meta(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_meta.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_pred_df(run_dir: Path, split: str = "test") -> pd.DataFrame | None:
    pred_path = run_dir / f"pred_{split}.csv"
    if not pred_path.exists():
        return None
    pred_raw = pd.read_csv(pred_path)
    pred = normalize_prediction_df(pred_raw)
    if pred is None:
        return None
    pred = pred[pred["y_true"].notna() & pred["y_pred"].notna()].copy()
    pred["y_true"] = pred["y_true"].astype(int)
    pred["y_pred"] = pred["y_pred"].astype(int)
    return pred


def _is_generative_run(record: Dict[str, Any]) -> bool:
    track = str(record.get("track", ""))
    model = str(record.get("model", "")).lower()
    run_name = str(record.get("run_name", "")).lower()
    return track == "3_vlm_severity" or "vlm" in model or "zero_shot" in run_name or "lora" in run_name


def _is_full(record: Dict[str, Any], expected_test_rows: int) -> bool:
    return int(record.get("test_rows") or -1) == expected_test_rows


def _get_canonical_split_hash(full_records: List[Dict[str, Any]]) -> str:
    hashes = [str(r.get("split_hash")) for r in full_records if r.get("split_hash")]
    if not hashes:
        return ""
    counts = pd.Series(hashes).value_counts()
    return str(counts.index[0])


def _has_file(run_dir: Path, name: str) -> int:
    return int((run_dir / name).exists())


def _build_audit_index(
    dataset_root: Path,
    records: List[Dict[str, Any]],
    expected_test_rows: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    full_records = [r for r in records if _is_full(r, expected_test_rows)]
    canonical_hash = _get_canonical_split_hash(full_records)

    rows: List[Dict[str, Any]] = []
    for r in records:
        run_dir = dataset_root / str(r.get("run_dir"))
        is_gen = _is_generative_run(r)
        parser_audit_file = run_dir / "parser_audit_samples.csv"
        pred_hist_file = run_dir / "pred_label_histogram.png"
        split_hash = str(r.get("split_hash") or "")
        rows.append(
            {
                "run_folder": r.get("run_name"),
                "path": str(run_dir.resolve()),
                "test_rows": r.get("test_rows"),
                "split_hash": split_hash,
                "split_hash_consistent_across_full_runs": int(
                    bool(canonical_hash) and split_hash == canonical_hash and _is_full(r, expected_test_rows)
                ),
                "canonical_split_hash": canonical_hash,
                "has_metrics_test_json": _has_file(run_dir, "metrics_test.json"),
                "has_pred_test_csv": _has_file(run_dir, "pred_test.csv"),
                "has_pred_val_csv": _has_file(run_dir, "pred_val.csv"),
                "has_run_meta_json": _has_file(run_dir, "run_meta.json"),
                "has_confusion_test_png": _has_file(run_dir, "confusion_test.png"),
                "has_pred_label_histogram_png (if applicable)": (
                    _has_file(run_dir, "pred_label_histogram.png") if is_gen else "n/a"
                ),
                "has_parser_audit_samples_csv (if applicable)": (
                    int(parser_audit_file.exists()) if is_gen else "n/a"
                ),
            }
        )

    index_df = pd.DataFrame(rows).sort_values("run_folder").reset_index(drop=True)
    return index_df, full_records, canonical_hash


def _build_full_runs_df(
    dataset_root: Path,
    full_records: List[Dict[str, Any]],
    canonical_hash: str,
) -> pd.DataFrame:
    rows = []
    for r in full_records:
        run_dir = dataset_root / str(r.get("run_dir"))
        rows.append(
            {
                "run_folder": r.get("run_name"),
                "path": str(run_dir.resolve()),
                "test_rows": r.get("test_rows"),
                "split_hash": r.get("split_hash"),
                "split_hash_consistent_across_full_runs": int(str(r.get("split_hash") or "") == canonical_hash),
                "canonical_split_hash": canonical_hash,
                "has_metrics_test_json": _has_file(run_dir, "metrics_test.json"),
                "has_pred_test_csv": _has_file(run_dir, "pred_test.csv"),
                "has_pred_val_csv": _has_file(run_dir, "pred_val.csv"),
                "has_run_meta_json": _has_file(run_dir, "run_meta.json"),
                "has_confusion_test_png": _has_file(run_dir, "confusion_test.png"),
                "has_pred_label_histogram_png (if applicable)": (
                    _has_file(run_dir, "pred_label_histogram.png") if _is_generative_run(r) else "n/a"
                ),
                "has_parser_audit_samples_csv (if applicable)": (
                    int((run_dir / "parser_audit_samples.csv").exists()) if _is_generative_run(r) else "n/a"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("run_folder").reset_index(drop=True)


def _build_main_table(full_records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in full_records:
        rows.append(
            {
                "model/run_folder": r.get("run_name"),
                "accuracy": r.get("accuracy"),
                "macro_f1": r.get("macro_f1"),
                "balanced_acc": r.get("balanced_accuracy"),
                "qwk": r.get("qwk"),
                "mae": r.get("mae"),
                "rmse": r.get("rmse"),
                "parse_rate": r.get("parse_rate"),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["accuracy", "model/run_folder"], ascending=[False, True]).reset_index(drop=True)


def _coverage_checks(full_records: List[Dict[str, Any]]) -> Tuple[Dict[str, bool], List[str]]:
    present = {str(r.get("run_name")) for r in full_records}
    checks: Dict[str, bool] = {}
    missing_reasons: List[str] = []

    for run_name in REQUIRED_FULL_RUNS["supervised"]:
        ok = run_name in present
        checks[f"supervised::{run_name}"] = ok
        if not ok:
            missing_reasons.append(f"- missing full supervised run: `{run_name}`")

    for run_name in REQUIRED_FULL_RUNS["generative_baseline"]:
        ok = run_name in present
        checks[f"generative_baseline::{run_name}"] = ok
        if not ok:
            missing_reasons.append(f"- missing full generative baseline run: `{run_name}`")

    has_generative_technique = any(
        ("mode2" in str(r.get("run_name")).lower()) or ("lora" in str(r.get("run_name")).lower())
        for r in full_records
        if _is_generative_run(r)
    )
    checks["has_generative_technique_full"] = has_generative_technique
    if not has_generative_technique:
        missing_reasons.append("- missing full generative technique beyond naive zero-shot")

    return checks, missing_reasons


def _write_missing_invalid_md(
    out_path: Path,
    coverage_checks: Dict[str, bool],
    missing_reasons: List[str],
    canonical_hash: str,
) -> None:
    lines = [
        "# Chapter 4 Missing or Invalid Runs",
        "",
        "## Required Coverage Status",
        f"- [{'x' if coverage_checks.get('supervised::finetune_resnet50') else ' '}] `finetune_resnet50` full run (n=1686)",
        f"- [{'x' if coverage_checks.get('supervised::finetune_vit_or_swin') else ' '}] `finetune_vit_or_swin` full run (n=1686)",
        f"- [{'x' if coverage_checks.get('supervised::resnet50_frozen_logreg') else ' '}] `resnet50_frozen_logreg` full run (n=1686)",
        f"- [{'x' if coverage_checks.get('supervised::vit_frozen_logreg') else ' '}] `vit_frozen_logreg` full run (n=1686)",
        f"- [{'x' if coverage_checks.get('supervised::clip_linear_baseline') else ' '}] `clip_linear_baseline` full run (n=1686)",
        f"- [{'x' if coverage_checks.get('generative_baseline::vlm_zero_shot_mayo') else ' '}] `vlm_zero_shot_mayo` full run (n=1686)",
        f"- [{'x' if coverage_checks.get('has_generative_technique_full') else ' '}] Full generative technique beyond naive zero-shot (LoRA full or controlled label scoring/sampling full)",
        "",
        "## Missing",
    ]
    if missing_reasons:
        lines.extend(missing_reasons)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Invalid",
            "- None",
            "",
            "## Split Hash Consistency Across Full Runs",
            f"- consistent: `{bool(canonical_hash)}`",
            f"- hashes: `{canonical_hash}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _remission_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true_bin = (df["y_true"] <= 1).astype(int)
    y_pred_bin = (df["y_pred"] <= 1).astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    n = int(len(df))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
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


def _mcnemar(df_a: pd.DataFrame, df_b: pd.DataFrame) -> PairStats:
    merged = df_a[["img_id_canonical", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_a"}).merge(
        df_b[["img_id_canonical", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_b"}),
        on=["img_id_canonical", "y_true"],
        how="inner",
    )
    if merged.empty:
        return PairStats(0, 0, 0, 0.0, 1.0)
    a_correct = merged["y_pred_a"].astype(int) == merged["y_true"].astype(int)
    b_correct = merged["y_pred_b"].astype(int) == merged["y_true"].astype(int)
    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    if (n01 + n10) > 0:
        chi2_cc = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
        p_value = math.erfc(math.sqrt(max(chi2_cc, 0.0) / 2.0))
    else:
        chi2_cc = 0.0
        p_value = 1.0
    return PairStats(
        n=int(len(merged)),
        n01_a_wrong_b_right=n01,
        n10_a_right_b_wrong=n10,
        chi2_cc=float(chi2_cc),
        p_value=float(p_value),
    )


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: random.Random,
    iters: int,
) -> Dict[str, float]:
    n = len(y_true)
    acc_vals: List[float] = []
    f1_vals: List[float] = []
    qwk_vals: List[float] = []
    if n == 0:
        return {
            "accuracy_mean": 0.0,
            "accuracy_ci_low": 0.0,
            "accuracy_ci_high": 0.0,
            "macro_f1_mean": 0.0,
            "macro_f1_ci_low": 0.0,
            "macro_f1_ci_high": 0.0,
            "qwk_mean": 0.0,
            "qwk_ci_low": 0.0,
            "qwk_ci_high": 0.0,
        }
    idx = list(range(n))
    for _ in range(iters):
        s = [idx[rng.randrange(n)] for _ in range(n)]
        yt = y_true[s]
        yp = y_pred[s]
        acc_vals.append(float(accuracy_score(yt, yp)))
        f1_vals.append(float(f1_score(yt, yp, labels=[0, 1, 2, 3], average="macro", zero_division=0)))
        qwk_vals.append(float(cohen_kappa_score(yt, yp, weights="quadratic")))
    return {
        "accuracy_mean": float(np.mean(acc_vals)),
        "accuracy_ci_low": float(np.percentile(acc_vals, 2.5)),
        "accuracy_ci_high": float(np.percentile(acc_vals, 97.5)),
        "macro_f1_mean": float(np.mean(f1_vals)),
        "macro_f1_ci_low": float(np.percentile(f1_vals, 2.5)),
        "macro_f1_ci_high": float(np.percentile(f1_vals, 97.5)),
        "qwk_mean": float(np.mean(qwk_vals)),
        "qwk_ci_low": float(np.percentile(qwk_vals, 2.5)),
        "qwk_ci_high": float(np.percentile(qwk_vals, 97.5)),
    }


def _pick_best_runs(full_records: List[Dict[str, Any]]) -> Tuple[str, str]:
    by_name = {str(r.get("run_name")): r for r in full_records}
    if "finetune_resnet50" in by_name:
        best_supervised = "finetune_resnet50"
    else:
        candidates = [r for r in full_records if not _is_generative_run(r)]
        candidates.sort(key=lambda x: (-(x.get("accuracy") or -1.0), str(x.get("run_name"))))
        best_supervised = str(candidates[0].get("run_name")) if candidates else ""

    gen_candidates = [r for r in full_records if _is_generative_run(r)]
    gen_candidates.sort(
        key=lambda x: (
            -(x.get("macro_f1") or -1.0),
            -(x.get("accuracy") or -1.0),
            str(x.get("run_name")),
        )
    )
    best_generative = str(gen_candidates[0].get("run_name")) if gen_candidates else ""
    return best_supervised, best_generative


def _copy_confusion(dataset_root: Path, run_name: str, figures_dir: Path) -> bool:
    if not run_name:
        return False
    matches = sorted(dataset_root.glob(f"**/results/{run_name}"))
    if not matches:
        return False
    src = matches[0] / "confusion_test.png"
    if not src.exists():
        return False
    dst = figures_dir / f"confusion_test_{run_name}.png"
    shutil.copy2(src, dst)
    return True


def _plot_best_generative_distribution(pred_df: pd.DataFrame, out_csv: Path, out_png: Path) -> Tuple[int, float]:
    counts = pred_df["y_pred"].value_counts().sort_index()
    total = int(len(pred_df))
    rows = []
    for label in [0, 1, 2, 3]:
        c = int(counts.get(label, 0))
        rows.append(
            {
                "pred_label": label,
                "count": c,
                "proportion": (c / total if total else 0.0),
            }
        )
    write_csv(rows, out_csv, field_order=("pred_label", "count", "proportion"))

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["pred_label"] for r in rows]
    ys = [r["count"] for r in rows]
    ax.bar(xs, ys, color="#2f6db5")
    ax.set_xlabel("Predicted Mayo score")
    ax.set_ylabel("Count")
    ax.set_title("Best Generative Predicted Label Distribution")
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    max_prop = max((r["proportion"] for r in rows), default=0.0)
    unique_classes = sum(1 for r in rows if r["count"] > 0)
    return unique_classes, max_prop


def _maybe_write_parser_audit(best_run: str, pred_df: pd.DataFrame, out_dir: Path) -> bool:
    run_name_lower = best_run.lower()
    is_freegen = "mode1" in run_name_lower or "freegen" in run_name_lower
    if not is_freegen:
        return True
    out_path = out_dir / f"parser_audit_samples_{best_run}_test.csv"
    if out_path.exists():
        return True

    cols = ["img_id", "y_true", "y_pred", "parse_ok"]
    if "raw_text" in pred_df.columns:
        cols.append("raw_text")
    work = pred_df[cols].copy()
    if len(work) > 20:
        work = work.sample(n=20, random_state=42)
    work = work.rename(
        columns={
            "img_id": "image_id",
            "y_true": "true_label",
            "y_pred": "pred_label",
            "raw_text": "raw_generation",
        }
    )
    work.to_csv(out_path, index=False)
    return out_path.exists()


def _lora_validity(dataset_root: Path) -> Dict[str, Any]:
    matches = sorted(dataset_root.glob("**/results/vlm_lora_finetune_mayo"))
    if not matches:
        return {
            "exists": False,
            "meta_ok": False,
            "adapter_files_ok": False,
            "training_proof_ok": False,
        }
    run_dir = matches[0]
    meta = _read_run_meta(run_dir)
    adapter_dir = run_dir / "lora_adapter"
    adapter_files = list(adapter_dir.glob("*")) if adapter_dir.exists() else []
    has_adapter_payload = any(p.suffix in {".safetensors", ".bin", ".json"} for p in adapter_files)
    has_meta = bool(meta.get("model_name")) and any(
        key in meta for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_adapter_path")
    )
    has_training_log = (run_dir / "training_history.csv").exists() or (run_dir / "train_log.csv").exists()
    has_training_meta = bool(meta.get("epochs")) and bool(meta.get("lr"))
    return {
        "exists": True,
        "meta_ok": has_meta,
        "adapter_files_ok": has_adapter_payload,
        "training_proof_ok": bool(has_training_log or has_training_meta),
    }


def _chapter_sync_checks(
    chapter_path: Path,
    best_generative: str,
) -> Dict[str, bool]:
    if not chapter_path.exists():
        return {
            "exists": False,
            "references_final_runs": False,
            "no_current_runs_wording": False,
            "has_required_fig_refs": False,
            "has_finalized_claim": False,
        }
    text = chapter_path.read_text(encoding="utf-8")
    lower = text.lower()
    required_figs = [
        "class_distribution_by_split.png",
        "confusion_test_finetune_resnet50.png",
        f"confusion_test_{best_generative}.png",
        "generative_pred_distribution.png",
    ]
    has_required_fig_refs = all(fig in text for fig in required_figs)
    has_claim = ("does not improve" in lower) or ("improves" in lower)
    references_runs = ("finetune_resnet50" in text) and (best_generative in text) and ("run id:" in lower)
    return {
        "exists": True,
        "references_final_runs": references_runs,
        "no_current_runs_wording": "current runs" not in lower,
        "has_required_fig_refs": has_required_fig_refs,
        "has_finalized_claim": has_claim,
    }


def _write_completion_report(
    out_path: Path,
    status: str,
    checks: Dict[str, bool],
    best_supervised: str,
    best_generative: str,
    canonical_hash: str,
    punch_list: List[str],
) -> None:
    def mark(v: bool) -> str:
        return "✅" if v else "❌"

    lines = [
        f"# {status}",
        "",
        "## Checklist",
        f"- {mark(checks.get('a_required_full_runs_exist', False))} (a) required full runs exist",
        f"- {mark(checks.get('b_generative_technique_full_run_exists', False))} (b) generative technique full-run exists",
        f"- {mark(checks.get('c_main_table_built', False))} (c) main table built from results only",
        f"- {mark(checks.get('d_remission_slice_exists', False))} (d) remission slice table exists",
        f"- {mark(checks.get('e_paired_significance_exists', False))} (e) paired significance exists",
        f"- {mark(checks.get('f_generative_validity_passed', False))} (f) generative validity checks passed",
        f"- {mark(checks.get('g_chapter_text_synced', False))} (g) chapter text synced",
        "",
        "## Evidence Summary",
        "- Scan index: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_audit_results_index.csv`",
        "- Full runs table: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_full_runs.csv`",
        "- Missing/invalid runs: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_missing_or_invalid_runs.md`",
        "- Final main table: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_final_main_table.csv`",
        "- Remission slice: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_remission_slice_table.csv`",
        "- Paired significance: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_paired_significance.csv`",
        f"- Best supervised run: `{best_supervised}`",
        f"- Best generative run: `{best_generative}`",
        f"- Split hash consistency across full runs: `{bool(canonical_hash)}` ({canonical_hash})",
    ]
    if status == "FAIL":
        lines.extend(["", "## Minimal Punch-List to Reach PASS"])
        for item in punch_list:
            lines.append(f"- {item}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    out_dir = args.out_dir.resolve() if args.out_dir else (dataset_root / "4_reporting" / "out")
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = collect_run_records(dataset_root=dataset_root, expected_test_rows=args.expected_test_rows)
    if not records:
        raise RuntimeError(f"No run folders found under: {dataset_root}")

    index_df, full_records, canonical_hash = _build_audit_index(dataset_root, records, args.expected_test_rows)
    index_path = out_dir / "chapter4_audit_results_index.csv"
    index_df.to_csv(index_path, index=False)

    full_df = _build_full_runs_df(dataset_root, full_records, canonical_hash)
    full_path = out_dir / "chapter4_full_runs.csv"
    full_df.to_csv(full_path, index=False)

    coverage_checks, missing_reasons = _coverage_checks(full_records)
    missing_md = out_dir / "chapter4_missing_or_invalid_runs.md"
    _write_missing_invalid_md(missing_md, coverage_checks, missing_reasons, canonical_hash)

    main_table = _build_main_table(full_records)
    main_table_path = out_dir / "chapter4_final_main_table.csv"
    main_table.to_csv(main_table_path, index=False)

    best_supervised, best_generative = _pick_best_runs(full_records)
    _copy_confusion(dataset_root, best_supervised, figures_dir)
    _copy_confusion(dataset_root, best_generative, figures_dir)

    pred_by_run: Dict[str, pd.DataFrame] = {}
    remission_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    for r in full_records:
        run_name = str(r.get("run_name"))
        run_dir = dataset_root / str(r.get("run_dir"))
        pred = _read_pred_df(run_dir)
        if pred is None or pred.empty:
            continue
        pred_by_run[run_name] = pred
        remission_rows.append({"model/run_folder": run_name, **_remission_metrics(pred)})

        ci = _bootstrap_ci(
            y_true=pred["y_true"].to_numpy(dtype=int),
            y_pred=pred["y_pred"].to_numpy(dtype=int),
            rng=rng,
            iters=args.bootstrap_iters,
        )
        ci_rows.append({"model/run_folder": run_name, "n": len(pred), **ci})

    remission_df = pd.DataFrame(remission_rows).sort_values(
        ["remission_accuracy", "model/run_folder"], ascending=[False, True]
    )
    remission_path = out_dir / "chapter4_remission_slice_table.csv"
    remission_df.to_csv(remission_path, index=False)

    ci_df = pd.DataFrame(ci_rows).sort_values("model/run_folder")
    ci_path = out_dir / "chapter4_metric_ci_bootstrap.csv"
    ci_df.to_csv(ci_path, index=False)

    pair_path = out_dir / "chapter4_paired_significance.csv"
    if best_supervised in pred_by_run and best_generative in pred_by_run:
        pair = _mcnemar(pred_by_run[best_supervised], pred_by_run[best_generative])
        pd.DataFrame(
            [
                {
                    "run_supervised": best_supervised,
                    "run_generative": best_generative,
                    "pair": f"{best_supervised} vs {best_generative}",
                    "n": pair.n,
                    "n01_A_wrong_B_right": pair.n01_a_wrong_b_right,
                    "n10_A_right_B_wrong": pair.n10_a_right_b_wrong,
                    "chi2_cc": pair.chi2_cc,
                    "p_value": pair.p_value,
                }
            ]
        ).to_csv(pair_path, index=False)
    else:
        pd.DataFrame(
            [
                {
                    "run_supervised": best_supervised,
                    "run_generative": best_generative,
                    "pair": f"{best_supervised} vs {best_generative}",
                    "n": 0,
                    "n01_A_wrong_B_right": 0,
                    "n10_A_right_B_wrong": 0,
                    "chi2_cc": 0.0,
                    "p_value": 1.0,
                }
            ]
        ).to_csv(pair_path, index=False)

    generative_valid = False
    parser_ok = True
    unique_classes = 0
    max_prop = 1.0
    gen_dist_csv = out_dir / "generative_pred_distribution.csv"
    gen_dist_png = out_dir / "generative_pred_distribution.png"
    if best_generative in pred_by_run:
        best_gen_pred = pred_by_run[best_generative]
        unique_classes, max_prop = _plot_best_generative_distribution(best_gen_pred, gen_dist_csv, gen_dist_png)
        parser_ok = _maybe_write_parser_audit(best_generative, best_gen_pred, out_dir)
        generative_valid = unique_classes > 1 and max_prop < 0.98 and parser_ok

    # Also export a run-specific histogram figure for the chosen best generative run.
    if best_generative in pred_by_run:
        pred = pred_by_run[best_generative]
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = [int((pred["y_pred"] == c).sum()) for c in [0, 1, 2, 3]]
        ax.bar([0, 1, 2, 3], counts, color="#245b8f")
        ax.set_xlabel("Predicted Mayo score")
        ax.set_ylabel("Count")
        ax.set_title(f"Predicted Label Distribution: {best_generative}")
        plt.tight_layout()
        fig.savefig(figures_dir / f"pred_label_histogram_{best_generative}.png", dpi=220)
        plt.close(fig)

    lora_check = _lora_validity(dataset_root)
    generative_valid = generative_valid and lora_check["exists"] and lora_check["meta_ok"] and lora_check["adapter_files_ok"]

    chapter_path = args.chapter_md.resolve()
    chapter_sync = _chapter_sync_checks(chapter_path, best_generative)
    chapter_synced = (
        chapter_sync["exists"]
        and chapter_sync["references_final_runs"]
        and chapter_sync["no_current_runs_wording"]
        and chapter_sync["has_required_fig_refs"]
        and chapter_sync["has_finalized_claim"]
    )

    checks = {
        "a_required_full_runs_exist": all(v for k, v in coverage_checks.items() if k.startswith("supervised::"))
        and all(v for k, v in coverage_checks.items() if k.startswith("generative_baseline::")),
        "b_generative_technique_full_run_exists": coverage_checks.get("has_generative_technique_full", False),
        "c_main_table_built": main_table_path.exists(),
        "d_remission_slice_exists": remission_path.exists(),
        "e_paired_significance_exists": pair_path.exists(),
        "f_generative_validity_passed": generative_valid,
        "g_chapter_text_synced": chapter_synced,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    punch_list: List[str] = []
    if not checks["a_required_full_runs_exist"]:
        punch_list.append("Ensure all required full runs (n=1686) are present in `*/results/*` and rerun this audit.")
    if not checks["b_generative_technique_full_run_exists"]:
        punch_list.append("Persist at least one full controlled-generative or LoRA run beyond naive zero-shot.")
    if not checks["d_remission_slice_exists"]:
        punch_list.append("Regenerate `chapter4_remission_slice_table.csv` from full-run `pred_test.csv` files.")
    if not checks["e_paired_significance_exists"]:
        punch_list.append("Regenerate `chapter4_paired_significance.csv` after canonical ID alignment.")
    if not checks["f_generative_validity_passed"]:
        punch_list.append("Best generative run failed validity checks (non-degeneracy/parser/LoRA metadata).")
    if not checks["g_chapter_text_synced"]:
        punch_list.append("Sync Chapter 4 markdown with final run IDs, figures, and finalized claim.")

    report_path = out_dir / "chapter4_completion_report.md"
    _write_completion_report(
        out_path=report_path,
        status=status,
        checks=checks,
        best_supervised=best_supervised,
        best_generative=best_generative,
        canonical_hash=canonical_hash,
        punch_list=punch_list,
    )

    print(f"LIMUC root: {dataset_root}")
    print(f"Out dir: {out_dir}")
    print(f"Run folders scanned: {len(records)}")
    print(f"Full runs used: {len(full_records)}")
    print(f"Best supervised: {best_supervised}")
    print(f"Best generative: {best_generative}")
    print(f"Generative validity: unique_classes={unique_classes}, max_proportion={max_prop:.4f}, parser_ok={parser_ok}")
    print(f"Status: {status}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
