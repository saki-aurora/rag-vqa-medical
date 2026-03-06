#!/usr/bin/env python3
"""Pass 8: stronger supervised backbone sweep for internal LIMUC push."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
        s = part.strip()
        if s:
            out.append(int(s))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No seeds parsed from input.")
    return uniq


def _cmd_to_text(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


@dataclass
class RunArtifact:
    config_name: str
    seed: int
    run_dir: Path
    log_path: Path
    run_meta: Dict[str, Any]
    metrics_val: Dict[str, Any]
    metrics_test: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _default_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "resnet50_ce_m",
            "backbone": "resnet50",
            "loss": "ce",
            "aug_strength": "medium",
            "image_size": 224,
            "resize_size": 256,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "epochs": 15,
            "scheduler": "cosine",
            "class_weighting": "balanced",
            "label_smoothing": 0.0,
        },
        {
            "name": "convnext_tiny_ce_m",
            "backbone": "convnext_tiny",
            "loss": "ce",
            "aug_strength": "medium",
            "image_size": 224,
            "resize_size": 256,
            "lr": 2e-4,
            "weight_decay": 2e-4,
            "batch_size": 16,
            "epochs": 15,
            "scheduler": "cosine",
            "class_weighting": "balanced",
            "label_smoothing": 0.0,
        },
        {
            "name": "convnext_tiny_focal_s",
            "backbone": "convnext_tiny",
            "loss": "focal",
            "focal_gamma": 1.5,
            "aug_strength": "strong",
            "image_size": 224,
            "resize_size": 256,
            "lr": 2e-4,
            "weight_decay": 2e-4,
            "batch_size": 16,
            "epochs": 15,
            "scheduler": "cosine",
            "class_weighting": "balanced",
            "label_smoothing": 0.0,
        },
        {
            "name": "swin_t_ce_m",
            "backbone": "swin_t",
            "loss": "ce",
            "aug_strength": "medium",
            "image_size": 224,
            "resize_size": 256,
            "lr": 1.5e-4,
            "weight_decay": 2e-4,
            "batch_size": 12,
            "epochs": 15,
            "scheduler": "cosine",
            "class_weighting": "balanced",
            "label_smoothing": 0.0,
        },
    ]


def parse_args() -> argparse.Namespace:
    default_root = find_limuc_root(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limuc-root", type=Path, default=default_root)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=default_root / "2_supervised_finetuning" / "train_supervised_backbone.py",
    )
    parser.add_argument("--scout-seed", type=int, default=11)
    parser.add_argument("--confirm-seeds", type=str, default="23,77")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mode", type=str, default="scout", choices=["scout", "full"])
    parser.add_argument("--tag", type=str, default="pass8_supervised_push")
    parser.add_argument("--out-dir", type=Path, default=default_root / "4_reporting" / "out")
    parser.add_argument("--results-dir", type=Path, default=default_root / "2_supervised_finetuning" / "results")
    parser.add_argument(
        "--configs-json",
        type=Path,
        default=None,
        help="Optional JSON list of config objects; falls back to built-in defaults.",
    )
    return parser.parse_args()


def _load_configs(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return _default_configs()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list at {path}")
    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "name" not in item:
            raise RuntimeError("Each config entry must include 'name'.")
        out.append(dict(item))
    if not out:
        raise RuntimeError("No usable configs found.")
    return out


def _build_run_name(config_name: str, seed: int) -> str:
    return f"{config_name}_seed{seed:03d}"


def _run_single(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    seed: int,
    logs_dir: Path,
) -> RunArtifact:
    run_name = _build_run_name(str(config["name"]), seed)
    run_dir = (args.results_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"

    cmd = [
        str(args.python),
        str(args.trainer_script.resolve()),
        "--limuc-root",
        str(args.limuc_root.resolve()),
        "--out-dir",
        str(run_dir),
        "--run-id",
        run_name,
        "--seed",
        str(seed),
        "--epochs",
        str(config.get("epochs", 15)),
        "--batch-size",
        str(config.get("batch_size", 16)),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(config.get("lr", 3e-4)),
        "--weight-decay",
        str(config.get("weight_decay", 1e-4)),
        "--min-lr",
        str(config.get("min_lr", 1e-6)),
        "--device",
        str(args.device),
        "--log-every",
        str(args.log_every),
        "--backbone",
        str(config.get("backbone", "resnet50")),
        "--loss",
        str(config.get("loss", "ce")),
        "--aug-strength",
        str(config.get("aug_strength", "medium")),
        "--image-size",
        str(config.get("image_size", 224)),
        "--resize-size",
        str(config.get("resize_size", 256)),
        "--class-weighting",
        str(config.get("class_weighting", "balanced")),
        "--scheduler",
        str(config.get("scheduler", "cosine")),
        "--label-smoothing",
        str(config.get("label_smoothing", 0.0)),
        "--early-stop-patience",
        str(args.early_stop_patience),
    ]
    if str(config.get("loss", "ce")) == "focal":
        cmd.extend(["--focal-gamma", str(config.get("focal_gamma", 2.0))])
    if args.max_samples and args.max_samples > 0:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.amp:
        cmd.append("--amp")

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"$ {_cmd_to_text(cmd)}\n\n")
        lf.flush()
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Run failed: {run_name} (seed={seed}) returncode={result.returncode}. Log: {log_path}")

    run_meta = _read_json(run_dir / "run_meta.json")
    metrics_val_payload = _read_json(run_dir / "metrics_val.json")
    metrics_test_payload = _read_json(run_dir / "metrics_test.json")
    metrics_val = metrics_val_payload.get("summary", metrics_val_payload)
    metrics_test = metrics_test_payload.get("summary", metrics_test_payload)

    return RunArtifact(
        config_name=str(config["name"]),
        seed=seed,
        run_dir=run_dir,
        log_path=log_path,
        run_meta=run_meta,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
    )


def _artifact_row(a: RunArtifact) -> Dict[str, Any]:
    return {
        "config_name": a.config_name,
        "seed": a.seed,
        "run_dir": str(a.run_dir),
        "log_path": str(a.log_path),
        "backbone": a.run_meta.get("backbone"),
        "loss": a.run_meta.get("loss"),
        "aug_strength": a.run_meta.get("aug_strength"),
        "epochs": a.run_meta.get("epochs"),
        "lr": a.run_meta.get("lr"),
        "weight_decay": a.run_meta.get("weight_decay"),
        "val_accuracy": a.metrics_val.get("accuracy"),
        "val_macro_f1": a.metrics_val.get("macro_f1"),
        "val_balanced_accuracy": a.metrics_val.get("balanced_accuracy"),
        "val_qwk": a.metrics_val.get("qwk"),
        "test_accuracy": a.metrics_test.get("accuracy"),
        "test_macro_f1": a.metrics_test.get("macro_f1"),
        "test_balanced_accuracy": a.metrics_test.get("balanced_accuracy"),
        "test_qwk": a.metrics_test.get("qwk"),
    }


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for name, g in rows.groupby("config_name", dropna=False):
        out_rows.append(
            {
                "config_name": name,
                "n_runs": int(len(g)),
                "test_accuracy_mean": float(g["test_accuracy"].mean()),
                "test_accuracy_std": float(g["test_accuracy"].std(ddof=0)),
                "test_macro_f1_mean": float(g["test_macro_f1"].mean()),
                "test_macro_f1_std": float(g["test_macro_f1"].std(ddof=0)),
                "test_qwk_mean": float(g["test_qwk"].mean()),
                "test_qwk_std": float(g["test_qwk"].std(ddof=0)),
                "val_qwk_mean": float(g["val_qwk"].mean()),
                "val_qwk_std": float(g["val_qwk"].std(ddof=0)),
                "best_single_test_qwk": float(g["test_qwk"].max()),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).sort_values(
        ["val_qwk_mean", "test_qwk_mean", "test_accuracy_mean"],
        ascending=[False, False, False],
    )


def main() -> None:
    args = parse_args()
    configs = _load_configs(args.configs_json.resolve() if args.configs_json else None)
    out_root = args.out_dir.resolve() / f"{args.tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[RunArtifact] = []
    scout_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        art = _run_single(args=args, config=cfg, seed=int(args.scout_seed), logs_dir=logs_dir)
        artifacts.append(art)
        scout_rows.append(_artifact_row(art))
        print(
            f"[scout] {cfg['name']} seed={args.scout_seed} "
            f"val_qwk={float(art.metrics_val.get('qwk', np.nan)):.4f} "
            f"test_qwk={float(art.metrics_test.get('qwk', np.nan)):.4f}"
        )

    scout_df = pd.DataFrame(scout_rows).sort_values(["val_qwk", "test_qwk"], ascending=[False, False]).reset_index(
        drop=True
    )
    scout_csv = out_root / "pass8_supervised_scout_runs.csv"
    scout_df.to_csv(scout_csv, index=False)

    top_cfg_names = scout_df["config_name"].head(max(int(args.top_k), 1)).tolist()
    top_cfgs = [c for c in configs if str(c["name"]) in set(top_cfg_names)]

    if args.mode == "full":
        confirm_seeds = _parse_seed_list(args.confirm_seeds)
        confirm_rows: List[Dict[str, Any]] = []
        for cfg in top_cfgs:
            for seed in confirm_seeds:
                art = _run_single(args=args, config=cfg, seed=int(seed), logs_dir=logs_dir)
                artifacts.append(art)
                confirm_rows.append(_artifact_row(art))
                print(
                    f"[confirm] {cfg['name']} seed={seed} "
                    f"val_qwk={float(art.metrics_val.get('qwk', np.nan)):.4f} "
                    f"test_qwk={float(art.metrics_test.get('qwk', np.nan)):.4f}"
                )
        if confirm_rows:
            pd.DataFrame(confirm_rows).to_csv(out_root / "pass8_supervised_confirm_runs.csv", index=False)

    all_rows = pd.DataFrame([_artifact_row(a) for a in artifacts])
    all_csv = out_root / "pass8_supervised_all_runs.csv"
    all_rows.to_csv(all_csv, index=False)

    agg_df = _aggregate(all_rows)
    agg_csv = out_root / "pass8_supervised_aggregate_by_config.csv"
    agg_df.to_csv(agg_csv, index=False)

    best_single = None
    if not all_rows.empty:
        best_single = all_rows.sort_values(["test_qwk", "test_accuracy"], ascending=[False, False]).iloc[0].to_dict()

    summary = {
        "generated_utc": _utc_now(),
        "mode": args.mode,
        "scout_seed": int(args.scout_seed),
        "confirm_seeds": _parse_seed_list(args.confirm_seeds) if args.mode == "full" else [],
        "top_k": int(args.top_k),
        "configs": [c["name"] for c in configs],
        "selected_top_configs": top_cfg_names,
        "best_single_run": best_single,
        "outputs": {
            "scout_csv": str(scout_csv),
            "all_runs_csv": str(all_csv),
            "aggregate_csv": str(agg_csv),
        },
    }
    summary_json = out_root / "pass8_supervised_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Pass 8 Supervised Push Report ({args.tag})",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Mode: `{args.mode}`",
        f"- Scout seed: `{args.scout_seed}`",
        f"- Selected top configs: `{', '.join(top_cfg_names)}`",
        "",
        "## Best Single Run",
        "",
    ]
    if best_single is None:
        md_lines.append("- none")
    else:
        md_lines.extend(
            [
                f"- config: `{best_single['config_name']}`",
                f"- seed: `{int(best_single['seed'])}`",
                f"- test_qwk: `{float(best_single['test_qwk']):.6f}`",
                f"- test_accuracy: `{float(best_single['test_accuracy']):.6f}`",
                f"- test_macro_f1: `{float(best_single['test_macro_f1']):.6f}`",
            ]
        )
    md_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{scout_csv}`",
            f"- `{all_csv}`",
            f"- `{agg_csv}`",
            f"- `{summary_json}`",
        ]
    )
    md_path = out_root / "pass8_supervised_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {scout_csv}")
    print(f"Wrote: {all_csv}")
    print(f"Wrote: {agg_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {md_path}")
    if best_single is not None:
        print(
            f"Best single: config={best_single['config_name']} seed={int(best_single['seed'])} "
            f"test_qwk={float(best_single['test_qwk']):.6f} test_acc={float(best_single['test_accuracy']):.6f}"
        )


if __name__ == "__main__":
    main()
