#!/usr/bin/env python3
"""Pass 5: run multi-seed supervised ResNet50 experiments and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            and (nested / "2_supervised_finetuning").exists()
            and (nested / "4_reporting").exists()
        ):
            return nested
        if (
            (p / "0_dataset_prep").exists()
            and (p / "2_supervised_finetuning").exists()
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
    if not uniq:
        raise ValueError("No seeds parsed from --seeds")
    return uniq


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


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


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


@dataclass
class SeedRun:
    seed: int
    run_dir: Path
    log_path: Path
    skipped_train: bool
    run_meta: Dict[str, Any]
    metrics_test: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return data


def _cmd_to_text(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def parse_args() -> argparse.Namespace:
    default_root = find_limuc_root(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limuc-root", type=Path, default=default_root)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=default_root / "2_supervised_finetuning" / "train_resnet50_finetune.py",
    )
    parser.add_argument("--seeds", type=str, default="11,23,42")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_root / "2_supervised_finetuning" / "results",
    )
    parser.add_argument("--run-prefix", type=str, default="finetune_resnet50_pass5_seed")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_root / "4_reporting" / "out",
    )
    parser.add_argument("--tag", type=str, default="pass5_supervised")
    return parser.parse_args()


def _train_or_load_seed_run(
    *,
    args: argparse.Namespace,
    seed: int,
    logs_dir: Path,
) -> SeedRun:
    run_dir = (args.results_dir / f"{args.run_prefix}{seed:03d}").resolve()
    log_path = (logs_dir / f"seed_{seed:03d}.log").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics_test.json"
    run_meta_path = run_dir / "run_meta.json"

    skipped_train = False
    should_train = not args.skip_train
    if metrics_path.exists() and run_meta_path.exists() and not args.force_retrain:
        should_train = False
        skipped_train = True

    if should_train:
        cmd = [
            str(Path(args.python).resolve()),
            str(args.trainer_script.resolve()),
            "--limuc-root",
            str(args.limuc_root.resolve()),
            "--out-dir",
            str(run_dir),
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--device",
            str(args.device),
            "--log-every",
            str(args.log_every),
        ]
        if args.max_samples and args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.amp:
            cmd.append("--amp")
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"$ {_cmd_to_text(cmd)}\n\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed for seed={seed} returncode={result.returncode}. "
                f"See log: {log_path}"
            )
    else:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_utc_now()}] skipped training; using existing artifacts in {run_dir}\n")

    run_meta = _read_json(run_meta_path)
    metrics_test = _read_json(metrics_path)
    return SeedRun(
        seed=seed,
        run_dir=run_dir,
        log_path=log_path,
        skipped_train=skipped_train,
        run_meta=run_meta,
        metrics_test=metrics_test,
    )


def _aggregate_metric_rows(
    rows: List[Dict[str, Any]],
    *,
    metric_keys: Sequence[str],
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in metric_keys:
        vals = [_safe_float(r.get(key)) for r in rows]
        arr = np.asarray([v for v in vals if v is not None], dtype=float)
        if arr.size == 0:
            out.append(
                {
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
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci_mean(
            arr.tolist(), n_boot=max(200, bootstrap_iters), rng=rng, alpha=0.05
        )
        out.append(
            {
                "metric": key,
                "n": int(arr.size),
                "mean": mean,
                "std": std,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return out


def _build_markdown_report(
    *,
    tag: str,
    run_rows: List[Dict[str, Any]],
    metric_summary_rows: List[Dict[str, Any]],
    per_class_rows: List[Dict[str, Any]],
    baseline_row: Dict[str, Any] | None,
    logs_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append(f"# {tag} Report")
    lines.append("")
    lines.append(f"- Generated UTC: `{_utc_now()}`")
    lines.append(f"- Number of seed runs: `{len(run_rows)}`")
    lines.append(f"- Logs directory: `{logs_dir}`")
    lines.append("")
    lines.append("## Seed Runs")
    lines.append("")
    for r in run_rows:
        lines.append(
            f"- seed `{r['seed']}`: run `{r['run_name']}` "
            f"(acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}, qwk={r['qwk']:.4f})"
        )
    lines.append("")
    lines.append("## Aggregate Metrics (mean ± std, 95% bootstrap CI)")
    lines.append("")
    for r in metric_summary_rows:
        if not math.isfinite(float(r["mean"])):
            continue
        lines.append(
            f"- {r['metric']}: {float(r['mean']):.4f} ± {float(r['std']):.4f} "
            f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]"
        )
    lines.append("")
    lines.append("## Per-Class Recall Aggregate")
    lines.append("")
    for r in per_class_rows:
        lines.append(
            f"- class {r['class_id']}: {float(r['mean_recall']):.4f} ± {float(r['std_recall']):.4f} "
            f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]"
        )
    if baseline_row is not None:
        lines.append("")
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append(
            "- baseline `finetune_resnet50`: "
            f"acc={float(baseline_row.get('accuracy', float('nan'))):.4f}, "
            f"macro_f1={float(baseline_row.get('macro_f1', float('nan'))):.4f}, "
            f"qwk={float(baseline_row.get('qwk', float('nan'))):.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def _plot_metric_errorbars(
    *,
    metric_rows: List[Dict[str, Any]],
    out_path: Path,
    metrics_to_plot: Sequence[str] = ("accuracy", "macro_f1", "balanced_accuracy", "qwk"),
) -> None:
    use = [r for r in metric_rows if str(r.get("metric")) in metrics_to_plot]
    if not use:
        return
    xs = np.arange(len(use))
    means = np.asarray([float(r["mean"]) for r in use], dtype=float)
    ci_lo = np.asarray([float(r["ci95_low"]) for r in use], dtype=float)
    ci_hi = np.asarray([float(r["ci95_high"]) for r in use], dtype=float)
    yerr = np.vstack([means - ci_lo, ci_hi - means])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(xs, means, color="#4e79a7", alpha=0.85)
    ax.errorbar(xs, means, yerr=yerr, fmt="none", ecolor="#1f2d3f", capsize=4, lw=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["metric"]) for r in use], rotation=15)
    ax.set_ylim(0.0, min(1.0, max(0.9, float(np.max(ci_hi) + 0.05))))
    ax.set_ylabel("Score")
    ax.set_title("Pass5 Supervised Multi-Seed Metrics (mean with 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / f"{args.tag}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seed_list(args.seeds)
    print(f"[pass5] seeds={seeds}")
    print(f"[pass5] limuc_root={limuc_root}")
    print(f"[pass5] trainer={args.trainer_script.resolve()}")
    print(f"[pass5] out_dir={out_dir}")

    seed_runs: List[SeedRun] = []
    for seed in seeds:
        print(f"[pass5] processing seed={seed} ...")
        run = _train_or_load_seed_run(args=args, seed=seed, logs_dir=logs_dir)
        seed_runs.append(run)
        print(
            f"[pass5] done seed={seed} skipped_train={run.skipped_train} "
            f"run_dir={run.run_dir}"
        )

    run_rows: List[Dict[str, Any]] = []
    per_class_seed_rows: List[Dict[str, Any]] = []
    agg_conf = np.zeros((4, 4), dtype=np.int64)

    for run in seed_runs:
        summary = run.metrics_test.get("summary", {})
        report = run.metrics_test.get("report", {})
        row: Dict[str, Any] = {
            "seed": run.seed,
            "run_name": run.run_dir.name,
            "run_dir": str(run.run_dir),
            "log_path": str(run.log_path),
            "run_id": run.run_meta.get("run_id"),
            "split_hash": run.run_meta.get("split_hash"),
            "epochs": run.run_meta.get("epochs"),
            "lr": run.run_meta.get("lr"),
            "weight_decay": run.run_meta.get("weight_decay"),
        }
        for key in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "qwk",
            "mae",
            "rmse",
            "spearman",
            "auroc_ovr",
            "ece",
        ]:
            row[key] = _safe_float(summary.get(key))
        run_rows.append(row)

        for class_id in ["0", "1", "2", "3"]:
            class_block = report.get(class_id, {})
            per_class_seed_rows.append(
                {
                    "seed": run.seed,
                    "run_name": run.run_dir.name,
                    "class_id": int(class_id),
                    "precision": _safe_float(class_block.get("precision")),
                    "recall": _safe_float(class_block.get("recall")),
                    "f1": _safe_float(class_block.get("f1-score")),
                    "support": _safe_float(class_block.get("support")),
                }
            )

        conf_path = run.run_dir / "confusion_test.npy"
        if conf_path.exists():
            arr = np.load(conf_path)
            if arr.shape == (4, 4):
                agg_conf += arr.astype(np.int64)

    run_rows.sort(key=lambda r: int(r["seed"]))
    _write_csv(run_rows, out_dir / "pass5_supervised_seed_runs.csv")
    _write_csv(per_class_seed_rows, out_dir / "pass5_supervised_per_class_seed_rows.csv")

    rng = np.random.default_rng(args.bootstrap_seed)
    metric_summary_rows = _aggregate_metric_rows(
        run_rows,
        metric_keys=[
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "qwk",
            "mae",
            "rmse",
            "auroc_ovr",
            "ece",
        ],
        bootstrap_iters=args.bootstrap_iters,
        rng=rng,
    )
    _write_csv(metric_summary_rows, out_dir / "pass5_supervised_metric_summary.csv")

    per_class_rows: List[Dict[str, Any]] = []
    pc_df = pd.DataFrame(per_class_seed_rows)
    for class_id in [0, 1, 2, 3]:
        sub = pc_df[pc_df["class_id"] == class_id].copy()
        vals = pd.to_numeric(sub["recall"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "n": 0,
                    "mean_recall": np.nan,
                    "std_recall": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                }
            )
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci_mean(
            vals.tolist(),
            n_boot=max(200, args.bootstrap_iters),
            rng=np.random.default_rng(args.bootstrap_seed + class_id),
            alpha=0.05,
        )
        per_class_rows.append(
            {
                "class_id": class_id,
                "n": int(vals.size),
                "mean_recall": mean,
                "std_recall": std,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            }
        )
    _write_csv(per_class_rows, out_dir / "pass5_supervised_per_class_recall_summary.csv")

    if int(agg_conf.sum()) > 0:
        np.save(out_dir / "pass5_supervised_confusion_aggregate.npy", agg_conf)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(agg_conf, cmap="Blues")
        ax.set_title("Pass5 Aggregated Test Confusion (seed-summed)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(agg_conf[i, j])}", ha="center", va="center", color="#112")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(out_dir / "pass5_supervised_confusion_aggregate.png", dpi=220)
        plt.close(fig)

    _plot_metric_errorbars(
        metric_rows=metric_summary_rows,
        out_path=out_dir / "figures" / "pass5_supervised_metric_ci.png",
    )

    baseline_path = (args.results_dir / "finetune_resnet50" / "metrics_test.json").resolve()
    baseline_row: Dict[str, Any] | None = None
    if baseline_path.exists():
        baseline_json = _read_json(baseline_path)
        baseline_row = baseline_json.get("summary", {}) if isinstance(baseline_json.get("summary"), dict) else None

    report = {
        "task": "pass5_supervised_multiseed",
        "generated_utc": _utc_now(),
        "limuc_root": str(limuc_root),
        "trainer_script": str(args.trainer_script.resolve()),
        "python": str(Path(args.python).resolve()),
        "seeds": seeds,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": bool(args.amp),
            "device": args.device,
            "max_samples": args.max_samples,
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "run_rows": run_rows,
        "metric_summary": metric_summary_rows,
        "per_class_recall_summary": per_class_rows,
        "baseline_summary_finetune_resnet50": baseline_row,
        "paths": {
            "seed_runs_csv": str((out_dir / "pass5_supervised_seed_runs.csv").resolve()),
            "metric_summary_csv": str((out_dir / "pass5_supervised_metric_summary.csv").resolve()),
            "per_class_seed_rows_csv": str((out_dir / "pass5_supervised_per_class_seed_rows.csv").resolve()),
            "per_class_recall_summary_csv": str((out_dir / "pass5_supervised_per_class_recall_summary.csv").resolve()),
            "aggregate_confusion_npy": str((out_dir / "pass5_supervised_confusion_aggregate.npy").resolve()),
            "aggregate_confusion_png": str((out_dir / "pass5_supervised_confusion_aggregate.png").resolve()),
            "metric_figure": str((out_dir / "figures" / "pass5_supervised_metric_ci.png").resolve()),
            "logs_dir": str(logs_dir.resolve()),
        },
    }

    report_json_path = out_dir / "pass5_supervised_multiseed_report.json"
    report_md_path = out_dir / "pass5_supervised_multiseed_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(
        _build_markdown_report(
            tag=args.tag,
            run_rows=run_rows,
            metric_summary_rows=metric_summary_rows,
            per_class_rows=per_class_rows,
            baseline_row=baseline_row,
            logs_dir=logs_dir,
        ),
        encoding="utf-8",
    )

    print("[pass5] complete")
    print(f"[pass5] report_json={report_json_path}")
    print(f"[pass5] report_md={report_md_path}")


if __name__ == "__main__":
    main()
