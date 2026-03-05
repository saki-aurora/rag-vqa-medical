#!/usr/bin/env python3
"""Pass 6: run multi-seed LoRA generative experiments and ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

try:
    from scipy.stats import binomtest as scipy_binomtest

    _HAS_SCIPY = True
except Exception:
    scipy_binomtest = None
    _HAS_SCIPY = False


METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "qwk",
    "mae",
    "rmse",
    "parse_rate",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_limuc_root(start: Path | None = None) -> Path:
    start_path = (start or Path.cwd()).resolve()
    script_root = Path(__file__).resolve().parents[1]
    candidates = [start_path] + list(start_path.parents) + [script_root] + list(script_root.parents)
    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        nested = p / "Prototyping_reformat" / "DatasetAnalysis" / "LIMUC"
        if (
            (nested / "0_dataset_prep").exists()
            and (nested / "3_vlm_severity").exists()
            and (nested / "4_reporting").exists()
        ):
            return nested
        if (
            (p / "0_dataset_prep").exists()
            and (p / "3_vlm_severity").exists()
            and (p / "4_reporting").exists()
        ):
            return p
    raise RuntimeError(f"Could not locate LIMUC root from start={start_path}")


def _parse_seed_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    uniq = sorted(set(out))
    return uniq


def _parse_str_list(raw: str) -> List[str]:
    out: List[str] = []
    for part in str(raw).split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return payload


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _cmd_to_text(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _bootstrap_ci_mean(
    values: Sequence[float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def _extract_summary(metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = metrics_payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return metrics_payload


def _normalize_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "y_true" in df.columns:
        rename_map["y_true"] = "true_label"
    if "y_pred" in df.columns:
        rename_map["y_pred"] = "pred_label"
    if "img_id" in df.columns:
        rename_map["img_id"] = "image_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"true_label", "pred_label"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Prediction CSV missing required columns: {sorted(required)}")

    if "image_id" not in df.columns:
        df["image_id"] = np.arange(len(df)).astype(str)
    if "parse_ok" not in df.columns:
        df["parse_ok"] = True

    out = df.copy()
    out["true_label"] = pd.to_numeric(out["true_label"], errors="coerce")
    out["pred_label"] = pd.to_numeric(out["pred_label"], errors="coerce")
    out["parse_ok"] = out["parse_ok"].astype(bool)
    out = out.dropna(subset=["true_label", "pred_label"]).reset_index(drop=True)
    out["true_label"] = out["true_label"].astype(int)
    out["pred_label"] = out["pred_label"].astype(int)
    out["image_id"] = out["image_id"].astype(str)
    return out


def _confusion_from_pred(pred_df: pd.DataFrame) -> np.ndarray:
    return confusion_matrix(
        pred_df["true_label"].to_numpy(dtype=int),
        pred_df["pred_label"].to_numpy(dtype=int),
        labels=[0, 1, 2, 3],
    ).astype(np.int64)


def _backfill_summary_metrics(summary: Dict[str, Any], pred_df: pd.DataFrame) -> Dict[str, Any]:
    """Fill missing summary metrics from prediction rows when source JSON is sparse."""

    out = dict(summary)
    y_true = pred_df["true_label"].to_numpy(dtype=int)
    y_pred = pred_df["pred_label"].to_numpy(dtype=int)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)

    if _safe_float(out.get("accuracy")) is None:
        out["accuracy"] = float(np.mean(y_true == y_pred))
    if _safe_float(out.get("macro_f1")) is None:
        out["macro_f1"] = float(report.get("macro avg", {}).get("f1-score", np.nan))
    if _safe_float(out.get("weighted_f1")) is None:
        out["weighted_f1"] = float(report.get("weighted avg", {}).get("f1-score", np.nan))
    if _safe_float(out.get("balanced_accuracy")) is None:
        recalls: List[float] = []
        for class_id in ["0", "1", "2", "3"]:
            recalls.append(float(report.get(class_id, {}).get("recall", 0.0)))
        out["balanced_accuracy"] = float(np.mean(recalls))
    if _safe_float(out.get("parse_rate")) is None and "parse_ok" in pred_df.columns:
        out["parse_rate"] = float(pred_df["parse_ok"].astype(bool).mean())
    return out


def _load_per_class_rows_from_csv(per_class_path: Path) -> List[Dict[str, Any]]:
    if not per_class_path.exists():
        return []
    df = pd.read_csv(per_class_path)
    if "class_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "class_id"})
        else:
            first = df.columns[0]
            df = df.rename(columns={first: "class_id"})
    rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        class_id = str(row.get("class_id"))
        if class_id not in {"0", "1", "2", "3"}:
            continue
        rows.append(
            {
                "class_id": int(class_id),
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1-score")),
                "support": _safe_float(row.get("support")),
            }
        )
    return rows


def _per_class_rows_from_pred(pred_df: pd.DataFrame) -> List[Dict[str, Any]]:
    report = classification_report(
        pred_df["true_label"].to_numpy(dtype=int),
        pred_df["pred_label"].to_numpy(dtype=int),
        labels=[0, 1, 2, 3],
        output_dict=True,
        zero_division=0,
    )
    rows: List[Dict[str, Any]] = []
    for class_id in [0, 1, 2, 3]:
        block = report.get(str(class_id), {})
        rows.append(
            {
                "class_id": class_id,
                "precision": _safe_float(block.get("precision")),
                "recall": _safe_float(block.get("recall")),
                "f1": _safe_float(block.get("f1-score")),
                "support": _safe_float(block.get("support")),
            }
        )
    return rows


def _ensure_lzma_shim(shim_dir: Path) -> Path:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "lzma.py"
    shim_code = """\
class LZMAError(Exception):
    pass

def _unsupported(*_args, **_kwargs):
    raise LZMAError("lzma is unavailable in this Python build")

class LZMAFile:
    def __init__(self, *_args, **_kwargs):
        _unsupported()
    def read(self, *_args, **_kwargs):
        _unsupported()
    def write(self, *_args, **_kwargs):
        _unsupported()
    def seek(self, *_args, **_kwargs):
        _unsupported()
    def tell(self, *_args, **_kwargs):
        _unsupported()
    def close(self, *_args, **_kwargs):
        _unsupported()

class LZMACompressor:
    def __init__(self, *_args, **_kwargs):
        _unsupported()
    def compress(self, *_args, **_kwargs):
        _unsupported()
    def flush(self, *_args, **_kwargs):
        _unsupported()

class LZMADecompressor:
    def __init__(self, *_args, **_kwargs):
        _unsupported()
    def decompress(self, *_args, **_kwargs):
        _unsupported()

open = _unsupported
compress = _unsupported
decompress = _unsupported
FORMAT_AUTO = 0
FORMAT_XZ = 1
FORMAT_ALONE = 2
FORMAT_RAW = 3
CHECK_NONE = 0
CHECK_CRC32 = 1
CHECK_CRC64 = 4
CHECK_SHA256 = 10
FILTER_LZMA1 = 0x4000000000000001
FILTER_LZMA2 = 0x21
FILTER_DELTA = 3
FILTER_X86 = 4
FILTER_IA64 = 5
FILTER_ARM = 6
FILTER_ARMTHUMB = 7
FILTER_POWERPC = 8
FILTER_SPARC = 9
MF_HC3 = 3
MF_HC4 = 4
MF_BT2 = 18
MF_BT3 = 19
MF_BT4 = 20
MODE_FAST = 1
MODE_NORMAL = 2
PRESET_DEFAULT = 6
PRESET_EXTREME = 2147483648
"""
    shim_path.write_text(shim_code, encoding="utf-8")
    return shim_path


def _subprocess_env(shim_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(shim_dir) if not old_pp else f"{shim_dir}:{old_pp}"
    return env


def _mcnemar_exact_pvalue(b: int, c: int) -> float:
    n = int(b + c)
    if n <= 0:
        return 1.0
    k = int(min(b, c))
    if _HAS_SCIPY and scipy_binomtest is not None:
        return float(scipy_binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)

    # Two-sided exact p-value from binomial distribution under p=0.5.
    probs = [math.comb(n, i) * (0.5**n) for i in range(n + 1)]
    p_obs = probs[k]
    p = sum(pi for pi in probs if pi <= p_obs + 1e-15)
    return float(min(1.0, p))


def _plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="#102a43")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_metric_compare(metric_rows: List[Dict[str, Any]], out_path: Path) -> None:
    lanes = ["lora_mode1_train", "lora_mode2_eval"]
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "qwk"]
    index = {(str(r["lane"]), str(r["metric"])): r for r in metric_rows}
    xs = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = {"lora_mode1_train": "#1f77b4", "lora_mode2_eval": "#ff7f0e"}
    labels = {"lora_mode1_train": "LoRA mode1", "lora_mode2_eval": "LoRA mode2"}

    for offset_idx, lane in enumerate(lanes):
        means: List[float] = []
        yerr_lo: List[float] = []
        yerr_hi: List[float] = []
        for metric in metrics:
            row = index.get((lane, metric))
            if row is None:
                means.append(np.nan)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            mean = float(row["mean"])
            lo = float(row["ci95_low"])
            hi = float(row["ci95_high"])
            means.append(mean)
            yerr_lo.append(max(0.0, mean - lo))
            yerr_hi.append(max(0.0, hi - mean))
        pos = xs + (offset_idx - 0.5) * width
        ax.bar(pos, means, width=width, color=colors[lane], alpha=0.9, label=labels[lane])
        ax.errorbar(pos, means, yerr=np.vstack([yerr_lo, yerr_hi]), fmt="none", ecolor="#111", capsize=3, lw=1.2)

    ax.set_xticks(xs)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Pass6 Generative Multi-Seed Metrics (mean with 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


@dataclass
class LoraRun:
    seed: int
    run_name: str
    run_dir: Path
    log_path: Path | None
    skipped_train: bool
    run_meta: Dict[str, Any]
    metrics_summary: Dict[str, Any]
    pred_df: pd.DataFrame
    confusion: np.ndarray
    per_class_rows: List[Dict[str, Any]]


@dataclass
class Mode2EvalRun:
    seed: int
    run_name: str
    run_dir: Path
    mode_dir: Path
    log_path: Path
    skipped_eval: bool
    run_meta: Dict[str, Any]
    metrics_summary: Dict[str, Any]
    pred_df: pd.DataFrame
    confusion: np.ndarray
    per_class_rows: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    default_root = find_limuc_root(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limuc-root", type=Path, default=default_root)
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument(
        "--train-script",
        type=Path,
        default=default_root / "3_vlm_severity" / "train_vlm_lora_mayo.py",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=default_root / "3_vlm_severity" / "controlled_vlm_mayo_eval.py",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=default_root / "0_dataset_prep" / "out" / "metadata" / "metadata_enriched.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_root / "3_vlm_severity" / "results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_root / "4_reporting" / "out",
    )
    parser.add_argument("--tag", type=str, default="pass6_generative")

    parser.add_argument("--new-seeds", type=str, default="11,23")
    parser.add_argument(
        "--existing-runs",
        type=str,
        default="vlm_lora_finetune_mayo_balanced_full_20260303",
        help="Comma-separated existing LoRA run folders under results-dir to include.",
    )
    parser.add_argument("--run-prefix", type=str, default="vlm_lora_finetune_mayo_balanced_pass6_seed")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument("--model-name", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-fast-processor", action="store_true")
    parser.add_argument(
        "--label-token-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through to train_vlm_lora_mayo.py to focus training loss on class token(s).",
    )
    parser.add_argument("--class-token-loss-weight", type=float, default=1.0)
    parser.add_argument("--template-token-loss-weight", type=float, default=0.10)

    parser.add_argument("--skip-mode2-eval", action="store_true")
    parser.add_argument("--force-reeval", action="store_true")
    parser.add_argument("--eval-run-prefix", type=str, default="vlm_lora_pass6_mode2_seed")
    parser.add_argument("--eval-log-every", type=int, default=100)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-processor-use-fast", action="store_true")
    parser.add_argument(
        "--eval-mode2-strategy",
        type=str,
        choices=["sequence_logprob", "next_token"],
        default="sequence_logprob",
        help="Controlled eval mode2 scoring strategy.",
    )

    parser.add_argument(
        "--zero-shot-mode1-run",
        type=str,
        default="vlm_zero_shot_mode1_freegen_from_results_20260301",
    )
    parser.add_argument(
        "--zero-shot-mode2-run",
        type=str,
        default="vlm_zero_shot_mode2_label_scoring_full_20260301",
    )
    parser.add_argument(
        "--zero-shot-mode2-sampling-run",
        type=str,
        default="vlm_zero_shot_mode2_label_sampling_full_20260302",
    )

    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-nonconverged-mode1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude non-converged mode1 training runs from mode1 aggregate metrics.",
    )
    parser.add_argument(
        "--mode1-min-qwk",
        type=float,
        default=0.20,
        help="Minimum mode1 QWK for a run to be treated as converged.",
    )
    parser.add_argument(
        "--mode1-min-pred-classes",
        type=int,
        default=2,
        help="Minimum number of unique predicted classes in mode1 test predictions.",
    )
    parser.add_argument(
        "--mode1-max-train-loss",
        type=float,
        default=3.0,
        help="Maximum allowed training loss for mode1 convergence gating (if training_summary.json exists).",
    )
    return parser.parse_args()


def _load_lora_run(run_dir: Path, seed_hint: int | None, log_path: Path | None, skipped_train: bool) -> LoraRun:
    run_meta = _read_json(run_dir / "run_meta.json")
    metrics_payload = _read_json(run_dir / "metrics_test.json")
    metrics_summary = _extract_summary(metrics_payload)

    pred_df = _normalize_pred_df(pd.read_csv(run_dir / "pred_test.csv"))
    metrics_summary = _backfill_summary_metrics(metrics_summary, pred_df)
    confusion = _confusion_from_pred(pred_df)

    per_class_rows = _load_per_class_rows_from_csv(run_dir / "per_class_test.csv")
    if not per_class_rows:
        per_class_rows = _per_class_rows_from_pred(pred_df)

    seed_val = seed_hint
    if seed_val is None:
        seed_meta = run_meta.get("seed")
        seed_val = int(seed_meta) if seed_meta is not None else -1

    return LoraRun(
        seed=int(seed_val),
        run_name=run_dir.name,
        run_dir=run_dir,
        log_path=log_path,
        skipped_train=skipped_train,
        run_meta=run_meta,
        metrics_summary=metrics_summary,
        pred_df=pred_df,
        confusion=confusion,
        per_class_rows=per_class_rows,
    )


def _train_or_load_seed_run(
    *,
    args: argparse.Namespace,
    seed: int,
    logs_dir: Path,
    shim_dir: Path,
) -> LoraRun:
    run_name = f"{args.run_prefix}{seed:03d}"
    run_dir = (args.results_dir / run_name).resolve()
    log_path = (logs_dir / f"seed_{seed:03d}_train.log").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics_test.json"
    run_meta_path = run_dir / "run_meta.json"

    skipped_train = False
    should_train = not args.skip_train
    if metrics_path.exists() and run_meta_path.exists() and not args.force_retrain:
        should_train = False
        skipped_train = True

    if should_train:
        if args.force_retrain and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd: List[str] = [
            str(Path(args.python).resolve()),
            str(args.train_script.resolve()),
            "--data-root",
            str(args.limuc_root.resolve()),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--model-name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--num-workers",
            str(args.num_workers),
            "--logging-steps",
            str(args.logging_steps),
            "--save-steps",
            str(args.save_steps),
            "--save-total-limit",
            str(args.save_total_limit),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
        ]
        if args.balanced_sampling:
            cmd.append("--balanced-sampling")
        if args.max_train_samples > 0:
            cmd.extend(["--max-train-samples", str(args.max_train_samples)])
        if args.max_val_samples > 0:
            cmd.extend(["--max-val-samples", str(args.max_val_samples)])
        if args.max_test_samples > 0:
            cmd.extend(["--max-test-samples", str(args.max_test_samples)])
        if args.force_cuda:
            cmd.append("--force-cuda")
        if args.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
        if args.use_fast_processor:
            cmd.append("--use-fast-processor")
        if args.label_token_only:
            cmd.append("--label-token-only")
        else:
            cmd.append("--no-label-token-only")
        cmd.extend(["--class-token-loss-weight", str(args.class_token_loss_weight)])
        cmd.extend(["--template-token-loss-weight", str(args.template_token_loss_weight)])

        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"$ {_cmd_to_text(cmd)}\n\n")
            lf.flush()
            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
                env=_subprocess_env(shim_dir),
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"LoRA training failed for seed={seed} returncode={result.returncode}. "
                f"See log: {log_path}"
            )
    else:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_utc_now()}] skipped training; using existing artifacts in {run_dir}\n")

    return _load_lora_run(run_dir=run_dir, seed_hint=seed, log_path=log_path, skipped_train=skipped_train)


def _load_existing_run(
    *,
    args: argparse.Namespace,
    run_name: str,
) -> LoraRun:
    run_dir = (args.results_dir / run_name).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Existing run folder not found: {run_dir}")
    return _load_lora_run(run_dir=run_dir, seed_hint=None, log_path=None, skipped_train=True)


def _build_mode1_qc_row(run: LoraRun, args: argparse.Namespace) -> Dict[str, Any]:
    qwk = _safe_float(run.metrics_summary.get("qwk"))
    acc = _safe_float(run.metrics_summary.get("accuracy"))
    parse_rate = _safe_float(run.metrics_summary.get("parse_rate"))
    pred_unique = int(run.pred_df["pred_label"].nunique()) if not run.pred_df.empty else 0

    train_loss: float | None = None
    train_summary_path = run.run_dir / "training_summary.json"
    if train_summary_path.exists():
        try:
            train_summary = _read_json(train_summary_path)
            train_loss = _safe_float(train_summary.get("train_loss"))
        except Exception:
            train_loss = None

    fail_reasons: List[str] = []
    if qwk is None or qwk < float(args.mode1_min_qwk):
        fail_reasons.append(f"qwk<{args.mode1_min_qwk}")
    if pred_unique < int(args.mode1_min_pred_classes):
        fail_reasons.append(f"pred_classes<{args.mode1_min_pred_classes}")
    if parse_rate is not None and parse_rate < 0.95:
        fail_reasons.append("parse_rate<0.95")
    if train_loss is not None and train_loss > float(args.mode1_max_train_loss):
        fail_reasons.append(f"train_loss>{args.mode1_max_train_loss}")

    return {
        "seed": int(run.seed),
        "run_name": str(run.run_name),
        "run_dir": str(run.run_dir),
        "accuracy": acc,
        "qwk": qwk,
        "parse_rate": parse_rate,
        "pred_unique_classes": int(pred_unique),
        "train_loss": train_loss,
        "qc_pass": len(fail_reasons) == 0,
        "qc_fail_reasons": ";".join(fail_reasons),
    }


def _run_or_load_mode2_eval(
    *,
    args: argparse.Namespace,
    lora_run: LoraRun,
    logs_dir: Path,
    shim_dir: Path,
) -> Mode2EvalRun:
    seed_for_eval = int(lora_run.seed if lora_run.seed >= 0 else 42)
    eval_run_name = f"{args.eval_run_prefix}{seed_for_eval:03d}"
    eval_root = (args.results_dir / eval_run_name).resolve()
    mode_dir = eval_root / "mode2_label_scoring"
    log_path = (logs_dir / f"seed_{seed_for_eval:03d}_mode2.log").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = mode_dir / "metrics_test.json"
    pred_path = mode_dir / "pred_test.csv"
    run_meta_path = eval_root / "run_meta.json"

    skipped_eval = False
    should_eval = not args.skip_mode2_eval
    if metrics_path.exists() and pred_path.exists() and run_meta_path.exists() and not args.force_reeval:
        should_eval = False
        skipped_eval = True

    adapter_dir = lora_run.run_dir / "lora_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory for run={lora_run.run_name}: {adapter_dir}")

    if should_eval:
        if args.force_reeval and eval_root.exists():
            shutil.rmtree(eval_root)
        eval_root.mkdir(parents=True, exist_ok=True)
        cmd: List[str] = [
            str(Path(args.python).resolve()),
            str(args.eval_script.resolve()),
            "--meta-csv",
            str(args.meta_csv.resolve()),
            "--split",
            "test",
            "--model-name",
            args.model_name,
            "--adapter-dir",
            str(adapter_dir.resolve()),
            "--mode",
            "mode2",
            "--mode2-strategy",
            str(args.eval_mode2_strategy),
            "--seed",
            str(seed_for_eval),
            "--run-name",
            eval_run_name,
            "--out-dir",
            str(args.results_dir.resolve()),
            "--log-every",
            str(args.eval_log_every),
        ]
        if args.eval_max_samples > 0:
            cmd.extend(["--max-samples", str(args.eval_max_samples)])
        if args.force_cuda:
            cmd.append("--force-cuda")
        if args.eval_processor_use_fast:
            cmd.append("--processor-use-fast")
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"$ {_cmd_to_text(cmd)}\n\n")
            lf.flush()
            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
                env=_subprocess_env(shim_dir),
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Mode2 controlled eval failed for seed={seed_for_eval} returncode={result.returncode}. "
                f"See log: {log_path}"
            )
    else:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_utc_now()}] skipped mode2 eval; using existing artifacts in {mode_dir}\n")

    run_meta = _read_json(run_meta_path)
    metrics_payload = _read_json(metrics_path)
    metrics_summary = _extract_summary(metrics_payload)
    pred_df = _normalize_pred_df(pd.read_csv(pred_path))
    metrics_summary = _backfill_summary_metrics(metrics_summary, pred_df)
    confusion = _confusion_from_pred(pred_df)
    per_class_rows = _per_class_rows_from_pred(pred_df)

    return Mode2EvalRun(
        seed=seed_for_eval,
        run_name=eval_run_name,
        run_dir=eval_root,
        mode_dir=mode_dir,
        log_path=log_path,
        skipped_eval=skipped_eval,
        run_meta=run_meta,
        metrics_summary=metrics_summary,
        pred_df=pred_df,
        confusion=confusion,
        per_class_rows=per_class_rows,
    )


def _collect_rows_for_lane(
    *,
    lane: str,
    runs: Sequence[LoraRun] | Sequence[Mode2EvalRun],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {
            "lane": lane,
            "seed": int(getattr(run, "seed")),
            "run_name": str(getattr(run, "run_name")),
            "run_dir": str(getattr(run, "run_dir")),
        }
        for key in METRIC_KEYS:
            row[key] = _safe_float(getattr(run, "metrics_summary").get(key))
        rows.append(row)
    rows.sort(key=lambda r: (str(r["lane"]), int(r["seed"]), str(r["run_name"])))
    return rows


def _aggregate_metric_rows(
    rows: List[Dict[str, Any]],
    *,
    metric_keys: Sequence[str],
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lanes = sorted(set(str(r.get("lane")) for r in rows))
    for lane in lanes:
        lane_rows = [r for r in rows if str(r.get("lane")) == lane]
        for key in metric_keys:
            vals = [_safe_float(r.get(key)) for r in lane_rows]
            arr = np.asarray([v for v in vals if v is not None], dtype=float)
            if arr.size == 0:
                out.append(
                    {
                        "lane": lane,
                        "metric": key,
                        "n": 0,
                        "mean": np.nan,
                        "std": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    }
                )
                continue
            ci_lo, ci_hi = _bootstrap_ci_mean(
                arr.tolist(),
                n_boot=max(200, bootstrap_iters),
                rng=np.random.default_rng(bootstrap_seed + hash((lane, key)) % 10000),
                alpha=0.05,
            )
            out.append(
                {
                    "lane": lane,
                    "metric": key,
                    "n": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            )
    return out


def _aggregate_per_class(per_class_seed_rows: List[Dict[str, Any]], *, bootstrap_iters: int, bootstrap_seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    df = pd.DataFrame(per_class_seed_rows)
    if df.empty:
        return out
    lanes = sorted(df["lane"].dropna().astype(str).unique().tolist())
    for lane in lanes:
        for class_id in [0, 1, 2, 3]:
            sub = df[(df["lane"] == lane) & (pd.to_numeric(df["class_id"], errors="coerce") == class_id)].copy()
            vals = pd.to_numeric(sub["recall"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                out.append(
                    {
                        "lane": lane,
                        "class_id": class_id,
                        "n": 0,
                        "mean_recall": np.nan,
                        "std_recall": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                    }
                )
                continue
            ci_lo, ci_hi = _bootstrap_ci_mean(
                vals.tolist(),
                n_boot=max(200, bootstrap_iters),
                rng=np.random.default_rng(bootstrap_seed + class_id + hash(lane) % 1000),
                alpha=0.05,
            )
            out.append(
                {
                    "lane": lane,
                    "class_id": class_id,
                    "n": int(vals.size),
                    "mean_recall": float(vals.mean()),
                    "std_recall": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                }
            )
    return out


def _load_baseline_metrics(results_dir: Path, run_name: str, mode_hint: str) -> Dict[str, Any] | None:
    if not run_name:
        return None
    run_dir = (results_dir / run_name).resolve()
    if not run_dir.exists():
        return None

    candidates = [
        run_dir / "metrics_test.json",
        run_dir / "mode1_free_generation" / "metrics_test.json",
        run_dir / "mode2_label_scoring" / "metrics_test.json",
    ]
    payload: Dict[str, Any] | None = None
    metrics_path: Path | None = None
    for c in candidates:
        if c.exists():
            payload = _read_json(c)
            metrics_path = c
            break
    if payload is None:
        return None

    summary = _extract_summary(payload)
    row: Dict[str, Any] = {
        "lane": "baseline_zero_shot",
        "variant": mode_hint,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path) if metrics_path else "",
    }
    for k in METRIC_KEYS:
        row[k] = _safe_float(summary.get(k))
    return row


def _build_markdown_report(
    *,
    tag: str,
    mode1_rows: List[Dict[str, Any]],
    mode2_rows: List[Dict[str, Any]],
    mode1_qc_rows: List[Dict[str, Any]],
    metric_summary_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
    mcnemar_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    logs_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append(f"# {tag} Report")
    lines.append("")
    lines.append(f"- Generated UTC: `{_utc_now()}`")
    lines.append(f"- LoRA mode1 seed runs: `{len(mode1_rows)}`")
    lines.append(f"- LoRA mode2 seed runs: `{len(mode2_rows)}`")
    lines.append(f"- Logs directory: `{logs_dir}`")
    lines.append("")
    if mode1_qc_rows:
        passed = [r for r in mode1_qc_rows if bool(r.get("qc_pass"))]
        failed = [r for r in mode1_qc_rows if not bool(r.get("qc_pass"))]
        lines.append("## Mode1 Convergence QC")
        lines.append("")
        lines.append(f"- QC pass count: `{len(passed)}`")
        lines.append(f"- QC fail count: `{len(failed)}`")
        for r in failed:
            lines.append(
                f"- excluded seed `{r['seed']}` run `{r['run_name']}`: reasons=`{r.get('qc_fail_reasons', '')}`"
            )
        lines.append("")
    lines.append("## LoRA Seed Runs (Mode1 from training)")
    lines.append("")
    for r in mode1_rows:
        lines.append(
            f"- seed `{r['seed']}` run `{r['run_name']}`: "
            f"acc={float(r.get('accuracy', np.nan)):.4f}, "
            f"macro_f1={float(r.get('macro_f1', np.nan)):.4f}, "
            f"qwk={float(r.get('qwk', np.nan)):.4f}"
        )
    lines.append("")
    lines.append("## LoRA Controlled Ablation (Mode2 label scoring)")
    lines.append("")
    for r in mode2_rows:
        lines.append(
            f"- seed `{r['seed']}` run `{r['run_name']}`: "
            f"acc={float(r.get('accuracy', np.nan)):.4f}, "
            f"macro_f1={float(r.get('macro_f1', np.nan)):.4f}, "
            f"qwk={float(r.get('qwk', np.nan)):.4f}"
        )
    lines.append("")
    lines.append("## Aggregate Metrics (mean +/- std, 95% bootstrap CI)")
    lines.append("")
    for lane in ["lora_mode1_train", "lora_mode2_eval"]:
        lines.append(f"- {lane}:")
        for r in metric_summary_rows:
            if str(r.get("lane")) != lane:
                continue
            mean = _safe_float(r.get("mean"))
            if mean is None:
                continue
            lines.append(
                f"  - {r['metric']}: {mean:.4f} +/- {float(r['std']):.4f} "
                f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]"
            )
    lines.append("")
    lines.append("## Mode2 - Mode1 Deltas by Seed")
    lines.append("")
    if delta_rows:
        for r in delta_rows:
            lines.append(
                f"- seed `{r['seed']}`: "
                f"d_acc={float(r.get('delta_accuracy', np.nan)):.4f}, "
                f"d_macro_f1={float(r.get('delta_macro_f1', np.nan)):.4f}, "
                f"d_qwk={float(r.get('delta_qwk', np.nan)):.4f}"
            )
    else:
        lines.append("- No overlapping seeds available for delta computation.")

    lines.append("")
    lines.append("## McNemar (Mode1 vs Mode2 by Seed)")
    lines.append("")
    if mcnemar_rows:
        for r in mcnemar_rows:
            lines.append(
                f"- seed `{r['seed']}`: b={r['b_mode1_correct_mode2_wrong']}, "
                f"c={r['c_mode1_wrong_mode2_correct']}, p={float(r['p_value_exact']):.6g}"
            )
    else:
        lines.append("- No overlapping predictions available for McNemar comparison.")

    lines.append("")
    lines.append("## Zero-shot Baselines")
    lines.append("")
    if baseline_rows:
        for r in baseline_rows:
            lines.append(
                f"- {r['variant']} ({r['run_name']}): "
                f"acc={float(r.get('accuracy', np.nan)):.4f}, "
                f"macro_f1={float(r.get('macro_f1', np.nan)):.4f}, "
                f"qwk={float(r.get('qwk', np.nan)):.4f}"
            )
    else:
        lines.append("- No baseline rows loaded.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.limuc_root = args.limuc_root.resolve()
    args.results_dir = args.results_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.meta_csv = args.meta_csv.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / f"{args.tag}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    new_seeds = _parse_seed_list(args.new_seeds)
    existing_runs = _parse_str_list(args.existing_runs)

    if not args.meta_csv.exists():
        raise FileNotFoundError(f"meta-csv not found: {args.meta_csv}")

    shim_dir = (args.out_dir / "_pyfix_lzma").resolve()
    _ensure_lzma_shim(shim_dir)

    print(f"[pass6] limuc_root={args.limuc_root}")
    print(f"[pass6] results_dir={args.results_dir}")
    print(f"[pass6] out_dir={args.out_dir}")
    print(f"[pass6] new_seeds={new_seeds}")
    print(f"[pass6] existing_runs={existing_runs}")

    lora_runs: List[LoraRun] = []
    for run_name in existing_runs:
        print(f"[pass6] loading existing run={run_name}")
        lora_runs.append(_load_existing_run(args=args, run_name=run_name))

    seen_seeds = {int(r.seed) for r in lora_runs if int(r.seed) >= 0}
    for seed in new_seeds:
        if seed in seen_seeds:
            print(f"[pass6] seed={seed} already covered by existing runs; skipping train launch")
            continue
        print(f"[pass6] processing training seed={seed}")
        run = _train_or_load_seed_run(args=args, seed=seed, logs_dir=logs_dir, shim_dir=shim_dir)
        lora_runs.append(run)
        print(f"[pass6] seed={seed} ready run_dir={run.run_dir}")

    if not lora_runs:
        raise RuntimeError("No LoRA runs available for Pass 6.")

    lora_runs = sorted(lora_runs, key=lambda r: (r.seed, r.run_name))
    mode1_qc_rows = [_build_mode1_qc_row(run, args) for run in lora_runs]
    mode1_runs_for_aggregate = list(lora_runs)
    if args.exclude_nonconverged_mode1:
        passed_names = {str(r["run_name"]) for r in mode1_qc_rows if bool(r.get("qc_pass"))}
        mode1_runs_for_aggregate = [run for run in lora_runs if str(run.run_name) in passed_names]
        if not mode1_runs_for_aggregate:
            print("[pass6] warning: all mode1 runs failed QC; falling back to including all mode1 runs")
            mode1_runs_for_aggregate = list(lora_runs)
    print(
        f"[pass6] mode1_qc pass={sum(int(bool(r.get('qc_pass'))) for r in mode1_qc_rows)} "
        f"fail={sum(int(not bool(r.get('qc_pass'))) for r in mode1_qc_rows)} "
        f"aggregate_n={len(mode1_runs_for_aggregate)}"
    )

    mode2_runs: List[Mode2EvalRun] = []
    for run in lora_runs:
        print(f"[pass6] processing mode2 eval for seed={run.seed} run={run.run_name}")
        mode2 = _run_or_load_mode2_eval(args=args, lora_run=run, logs_dir=logs_dir, shim_dir=shim_dir)
        mode2_runs.append(mode2)
        print(f"[pass6] mode2 seed={mode2.seed} ready mode_dir={mode2.mode_dir}")

    mode1_rows = _collect_rows_for_lane(lane="lora_mode1_train", runs=mode1_runs_for_aggregate)
    mode2_rows = _collect_rows_for_lane(lane="lora_mode2_eval", runs=mode2_runs)
    all_rows = mode1_rows + mode2_rows

    per_class_seed_rows: List[Dict[str, Any]] = []
    for run in mode1_runs_for_aggregate:
        for row in run.per_class_rows:
            per_class_seed_rows.append(
                {
                    "lane": "lora_mode1_train",
                    "seed": run.seed,
                    "run_name": run.run_name,
                    **row,
                }
            )
    for run in mode2_runs:
        for row in run.per_class_rows:
            per_class_seed_rows.append(
                {
                    "lane": "lora_mode2_eval",
                    "seed": run.seed,
                    "run_name": run.run_name,
                    **row,
                }
            )

    metric_summary_rows = _aggregate_metric_rows(
        all_rows,
        metric_keys=METRIC_KEYS,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )
    per_class_summary_rows = _aggregate_per_class(
        per_class_seed_rows,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    mode1_df = pd.DataFrame(mode1_rows)
    mode2_df = pd.DataFrame(mode2_rows)
    delta_rows: List[Dict[str, Any]] = []
    if not mode1_df.empty and not mode2_df.empty:
        merged = mode1_df.merge(mode2_df, on="seed", suffixes=("_mode1", "_mode2"))
        for _, row in merged.iterrows():
            out: Dict[str, Any] = {
                "seed": int(row["seed"]),
                "run_name_mode1": row["run_name_mode1"],
                "run_name_mode2": row["run_name_mode2"],
            }
            for key in METRIC_KEYS:
                v1 = _safe_float(row.get(f"{key}_mode1"))
                v2 = _safe_float(row.get(f"{key}_mode2"))
                out[f"mode1_{key}"] = v1
                out[f"mode2_{key}"] = v2
                out[f"delta_{key}"] = None if (v1 is None or v2 is None) else float(v2 - v1)
            delta_rows.append(out)

    mcnemar_rows: List[Dict[str, Any]] = []
    mode2_by_seed = {int(r.seed): r for r in mode2_runs}
    for run in mode1_runs_for_aggregate:
        mode2 = mode2_by_seed.get(int(run.seed))
        if mode2 is None:
            continue
        p1 = run.pred_df[["image_id", "true_label", "pred_label"]].copy().rename(columns={"pred_label": "pred1"})
        p2 = mode2.pred_df[["image_id", "true_label", "pred_label"]].copy().rename(columns={"pred_label": "pred2"})
        m = p1.merge(p2, on=["image_id", "true_label"], how="inner")
        if m.empty:
            continue
        c1 = m["pred1"].astype(int).to_numpy() == m["true_label"].astype(int).to_numpy()
        c2 = m["pred2"].astype(int).to_numpy() == m["true_label"].astype(int).to_numpy()
        b = int(np.sum(c1 & ~c2))
        c = int(np.sum(~c1 & c2))
        pval = _mcnemar_exact_pvalue(b, c)
        chi2_cc = float(((abs(b - c) - 1.0) ** 2) / max(b + c, 1))
        mcnemar_rows.append(
            {
                "seed": int(run.seed),
                "n": int(len(m)),
                "b_mode1_correct_mode2_wrong": b,
                "c_mode1_wrong_mode2_correct": c,
                "chi2_cc": chi2_cc,
                "p_value_exact": pval,
            }
        )

    baseline_rows: List[Dict[str, Any]] = []
    for run_name, variant in [
        (args.zero_shot_mode1_run, "zero_shot_mode1"),
        (args.zero_shot_mode2_run, "zero_shot_mode2_scoring"),
        (args.zero_shot_mode2_sampling_run, "zero_shot_mode2_sampling"),
    ]:
        row = _load_baseline_metrics(args.results_dir, run_name, variant)
        if row is not None:
            baseline_rows.append(row)

    ablation_rows: List[Dict[str, Any]] = []
    for r in mode1_rows:
        ablation_rows.append({"family": "lora", "variant": "mode1_train", **r})
    for r in mode2_rows:
        ablation_rows.append({"family": "lora", "variant": "mode2_label_scoring", **r})
    for r in baseline_rows:
        out = dict(r)
        out["family"] = "zero_shot"
        out["seed"] = None
        ablation_rows.append(out)

    agg_mode1 = np.zeros((4, 4), dtype=np.int64)
    agg_mode2 = np.zeros((4, 4), dtype=np.int64)
    for run in lora_runs:
        if run.confusion.shape == (4, 4):
            agg_mode1 += run.confusion.astype(np.int64)
    for run in mode2_runs:
        if run.confusion.shape == (4, 4):
            agg_mode2 += run.confusion.astype(np.int64)

    # Persist artifacts
    _write_csv(mode1_rows, args.out_dir / "pass6_generative_lora_mode1_seed_runs.csv")
    _write_csv(mode2_rows, args.out_dir / "pass6_generative_lora_mode2_seed_runs.csv")
    _write_csv(mode1_qc_rows, args.out_dir / "pass6_generative_mode1_qc.csv")
    _write_csv(metric_summary_rows, args.out_dir / "pass6_generative_metric_summary.csv")
    _write_csv(per_class_seed_rows, args.out_dir / "pass6_generative_per_class_seed_rows.csv")
    _write_csv(per_class_summary_rows, args.out_dir / "pass6_generative_per_class_recall_summary.csv")
    _write_csv(delta_rows, args.out_dir / "pass6_generative_mode2_minus_mode1_by_seed.csv")
    _write_csv(mcnemar_rows, args.out_dir / "pass6_generative_mcnemar_mode1_vs_mode2.csv")
    _write_csv(baseline_rows, args.out_dir / "pass6_generative_zero_shot_baselines.csv")
    _write_csv(ablation_rows, args.out_dir / "pass6_generative_ablation_table.csv")

    np.save(args.out_dir / "pass6_generative_confusion_aggregate_mode1.npy", agg_mode1)
    np.save(args.out_dir / "pass6_generative_confusion_aggregate_mode2.npy", agg_mode2)
    _plot_confusion(
        agg_mode1,
        title="Pass6 Aggregated Confusion - LoRA Mode1 (train run outputs)",
        out_path=args.out_dir / "pass6_generative_confusion_aggregate_mode1.png",
    )
    _plot_confusion(
        agg_mode2,
        title="Pass6 Aggregated Confusion - LoRA Mode2 (controlled eval)",
        out_path=args.out_dir / "pass6_generative_confusion_aggregate_mode2.png",
    )
    _plot_metric_compare(
        metric_rows=metric_summary_rows,
        out_path=args.out_dir / "figures" / "pass6_generative_metric_ci.png",
    )

    report = {
        "task": "pass6_generative_multiseed_ablations",
        "generated_utc": _utc_now(),
        "limuc_root": str(args.limuc_root),
        "results_dir": str(args.results_dir),
        "out_dir": str(args.out_dir),
        "python": str(Path(args.python).resolve()),
        "train_script": str(args.train_script.resolve()),
        "eval_script": str(args.eval_script.resolve()),
        "meta_csv": str(args.meta_csv.resolve()),
        "config": {
            "new_seeds": new_seeds,
            "existing_runs": existing_runs,
            "skip_train": bool(args.skip_train),
            "force_retrain": bool(args.force_retrain),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "balanced_sampling": bool(args.balanced_sampling),
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "label_token_only": bool(args.label_token_only),
            "class_token_loss_weight": float(args.class_token_loss_weight),
            "template_token_loss_weight": float(args.template_token_loss_weight),
            "skip_mode2_eval": bool(args.skip_mode2_eval),
            "force_reeval": bool(args.force_reeval),
            "eval_mode2_strategy": str(args.eval_mode2_strategy),
            "eval_max_samples": args.eval_max_samples,
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "exclude_nonconverged_mode1": bool(args.exclude_nonconverged_mode1),
            "mode1_min_qwk": float(args.mode1_min_qwk),
            "mode1_min_pred_classes": int(args.mode1_min_pred_classes),
            "mode1_max_train_loss": float(args.mode1_max_train_loss),
        },
        "mode1_qc_rows": mode1_qc_rows,
        "mode1_seed_rows": mode1_rows,
        "mode2_seed_rows": mode2_rows,
        "metric_summary": metric_summary_rows,
        "per_class_seed_rows": per_class_seed_rows,
        "per_class_recall_summary": per_class_summary_rows,
        "mode2_minus_mode1": delta_rows,
        "mcnemar_mode1_vs_mode2": mcnemar_rows,
        "zero_shot_baselines": baseline_rows,
        "paths": {
            "mode1_seed_runs_csv": str((args.out_dir / "pass6_generative_lora_mode1_seed_runs.csv").resolve()),
            "mode2_seed_runs_csv": str((args.out_dir / "pass6_generative_lora_mode2_seed_runs.csv").resolve()),
            "mode1_qc_csv": str((args.out_dir / "pass6_generative_mode1_qc.csv").resolve()),
            "metric_summary_csv": str((args.out_dir / "pass6_generative_metric_summary.csv").resolve()),
            "per_class_seed_rows_csv": str((args.out_dir / "pass6_generative_per_class_seed_rows.csv").resolve()),
            "per_class_recall_summary_csv": str((args.out_dir / "pass6_generative_per_class_recall_summary.csv").resolve()),
            "delta_csv": str((args.out_dir / "pass6_generative_mode2_minus_mode1_by_seed.csv").resolve()),
            "mcnemar_csv": str((args.out_dir / "pass6_generative_mcnemar_mode1_vs_mode2.csv").resolve()),
            "baseline_csv": str((args.out_dir / "pass6_generative_zero_shot_baselines.csv").resolve()),
            "ablation_table_csv": str((args.out_dir / "pass6_generative_ablation_table.csv").resolve()),
            "confusion_mode1_npy": str((args.out_dir / "pass6_generative_confusion_aggregate_mode1.npy").resolve()),
            "confusion_mode1_png": str((args.out_dir / "pass6_generative_confusion_aggregate_mode1.png").resolve()),
            "confusion_mode2_npy": str((args.out_dir / "pass6_generative_confusion_aggregate_mode2.npy").resolve()),
            "confusion_mode2_png": str((args.out_dir / "pass6_generative_confusion_aggregate_mode2.png").resolve()),
            "metric_figure": str((args.out_dir / "figures" / "pass6_generative_metric_ci.png").resolve()),
            "logs_dir": str(logs_dir.resolve()),
        },
    }

    report_json_path = args.out_dir / "pass6_generative_multiseed_report.json"
    report_md_path = args.out_dir / "pass6_generative_multiseed_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(
        _build_markdown_report(
            tag=args.tag,
            mode1_rows=mode1_rows,
            mode2_rows=mode2_rows,
            mode1_qc_rows=mode1_qc_rows,
            metric_summary_rows=metric_summary_rows,
            delta_rows=delta_rows,
            mcnemar_rows=mcnemar_rows,
            baseline_rows=baseline_rows,
            logs_dir=logs_dir,
        ),
        encoding="utf-8",
    )

    print("[pass6] complete")
    print(f"[pass6] report_json={report_json_path}")
    print(f"[pass6] report_md={report_md_path}")


if __name__ == "__main__":
    main()
