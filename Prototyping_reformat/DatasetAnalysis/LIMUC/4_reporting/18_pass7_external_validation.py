#!/usr/bin/env python3
"""Pass 7: external/domain-shift validation for best supervised and generative runs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
            (nested / "2_supervised_finetuning").exists()
            and (nested / "3_vlm_severity").exists()
            and (nested / "4_reporting").exists()
        ):
            return nested
        if (
            (p / "2_supervised_finetuning").exists()
            and (p / "3_vlm_severity").exists()
            and (p / "4_reporting").exists()
        ):
            return p
    raise RuntimeError(f"Could not locate LIMUC root from start={start_path}")


def _extract_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return data


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


def _cmd_text(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _run_cmd(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"$ {_cmd_text(cmd)}\n\n")
        lf.flush()
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (code={result.returncode}): {_cmd_text(cmd)}; see {log_path}")


def _find_seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)", str(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    default_root = find_limuc_root(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limuc-root", type=Path, default=default_root)
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument("--meta-csv", type=Path, required=True, help="External metadata CSV with split/image_path/label_id.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tag", type=str, default="pass7_external")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_root / "4_reporting" / "out",
        help="Base output directory; script writes into <out-dir>/<tag>/",
    )

    parser.add_argument(
        "--resnet-eval-script",
        type=Path,
        default=default_root / "2_supervised_finetuning" / "eval_resnet50_checkpoint.py",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=default_root / "2_supervised_finetuning" / "results" / "finetune_resnet50" / "best_resnet50.pt",
    )
    parser.add_argument("--resnet-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--vlm-eval-script",
        type=Path,
        default=default_root / "3_vlm_severity" / "controlled_vlm_mayo_eval.py",
    )
    parser.add_argument(
        "--vlm-run-dir",
        type=Path,
        default=default_root / "3_vlm_severity" / "results" / "vlm_lora_objfix_b200_seed077",
    )
    parser.add_argument("--model-name", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--eval-mode2-strategy", type=str, default="sequence_logprob", choices=["sequence_logprob", "next_token"])
    parser.add_argument("--eval-log-every", type=int, default=100)

    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    meta_csv = args.meta_csv.resolve()
    out_base = args.out_dir.resolve() / args.tag
    out_base.mkdir(parents=True, exist_ok=True)
    logs_dir = out_base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not meta_csv.exists():
        raise FileNotFoundError(f"meta-csv not found: {meta_csv}")

    print(f"[pass7] limuc_root={limuc_root}")
    print(f"[pass7] meta_csv={meta_csv}")
    print(f"[pass7] split={args.split}")
    print(f"[pass7] out_base={out_base}")

    resnet_eval_out = out_base / "resnet50_external_eval"
    vlm_out_parent = out_base / "vlm_external_eval"
    vlm_run_name = f"{args.tag}_vlm_{args.split}"
    vlm_run_root = vlm_out_parent / vlm_run_name

    executed_cmds: List[Dict[str, str]] = []

    if not args.skip_resnet:
        resnet_cmd: List[str] = [
            str(Path(args.python).expanduser()),
            str(args.resnet_eval_script.resolve()),
            "--meta-csv",
            str(meta_csv),
            "--split",
            str(args.split),
            "--checkpoint",
            str(args.resnet_checkpoint.resolve()),
            "--out-dir",
            str(resnet_eval_out),
            "--limuc-root",
            str(limuc_root),
            "--batch-size",
            str(args.resnet_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--run-name",
            "resnet50_external_eval",
        ]
        if args.max_samples > 0:
            resnet_cmd.extend(["--max-samples", str(args.max_samples)])
        if args.force_cuda:
            resnet_cmd.extend(["--device", "cuda"])
        resnet_log = logs_dir / "resnet_eval.log"
        _run_cmd(resnet_cmd, resnet_log)
        executed_cmds.append({"stage": "resnet_eval", "command": _cmd_text(resnet_cmd), "log": str(resnet_log)})

    if not args.skip_vlm:
        adapter_dir = args.vlm_run_dir.resolve() / "lora_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"adapter-dir not found: {adapter_dir}")
        vlm_cmd: List[str] = [
            str(Path(args.python).expanduser()),
            str(args.vlm_eval_script.resolve()),
            "--meta-csv",
            str(meta_csv),
            "--split",
            str(args.split),
            "--model-name",
            str(args.model_name),
            "--adapter-dir",
            str(adapter_dir),
            "--mode",
            "both",
            "--mode2-strategy",
            str(args.eval_mode2_strategy),
            "--seed",
            str(args.seed),
            "--run-name",
            vlm_run_name,
            "--out-dir",
            str(vlm_out_parent),
            "--log-every",
            str(args.eval_log_every),
        ]
        if args.max_samples > 0:
            vlm_cmd.extend(["--max-samples", str(args.max_samples)])
        if args.force_cuda:
            vlm_cmd.append("--force-cuda")
        vlm_log = logs_dir / "vlm_eval.log"
        _run_cmd(vlm_cmd, vlm_log)
        executed_cmds.append({"stage": "vlm_eval", "command": _cmd_text(vlm_cmd), "log": str(vlm_log)})

    # Internal references
    internal_rows: Dict[str, Dict[str, Any]] = {}

    internal_resnet_metrics = _extract_summary(
        _read_json(limuc_root / "2_supervised_finetuning" / "results" / "finetune_resnet50" / "metrics_test.json")
    )
    internal_rows["resnet50_supervised"] = internal_resnet_metrics

    seed_name = args.vlm_run_dir.name
    seed_id = _find_seed_from_name(seed_name)
    mode1_csv = limuc_root / "4_reporting" / "out" / "pass6_generative_lora_mode1_seed_runs.csv"
    mode2_csv = limuc_root / "4_reporting" / "out" / "pass6_generative_lora_mode2_seed_runs.csv"

    mode1_df = pd.read_csv(mode1_csv) if mode1_csv.exists() else pd.DataFrame()
    mode2_df = pd.read_csv(mode2_csv) if mode2_csv.exists() else pd.DataFrame()
    m1_row = None
    if not mode1_df.empty and "run_name" in mode1_df.columns:
        hit = mode1_df[mode1_df["run_name"].astype(str) == seed_name]
        if not hit.empty:
            m1_row = hit.iloc[0].to_dict()
    if m1_row is None:
        m1_row = _extract_summary(_read_json(args.vlm_run_dir.resolve() / "metrics_test.json"))

    m2_row = None
    if seed_id is not None and not mode2_df.empty and "seed" in mode2_df.columns:
        hit2 = mode2_df[pd.to_numeric(mode2_df["seed"], errors="coerce") == int(seed_id)]
        if not hit2.empty:
            m2_row = hit2.iloc[0].to_dict()
    if m2_row is None and seed_id is not None:
        m2_path = limuc_root / "3_vlm_severity" / "results" / f"vlm_lora_pass6_mode2_seed{seed_id:03d}" / "mode2_label_scoring" / "metrics_test.json"
        if m2_path.exists():
            m2_row = _extract_summary(_read_json(m2_path))
    if m2_row is None:
        m2_row = {}

    internal_rows["vlm_lora_mode1"] = m1_row
    internal_rows["vlm_lora_mode2"] = m2_row

    # External metrics
    external_rows: Dict[str, Dict[str, Any]] = {}
    if not args.skip_resnet:
        ext_resnet_path = resnet_eval_out / f"metrics_{args.split}.json"
        external_rows["resnet50_supervised"] = _extract_summary(_read_json(ext_resnet_path))
    if not args.skip_vlm:
        ext_m1 = _extract_summary(_read_json(vlm_run_root / "mode1_free_generation" / "metrics_test.json"))
        ext_m2 = _extract_summary(_read_json(vlm_run_root / "mode2_label_scoring" / "metrics_test.json"))
        external_rows["vlm_lora_mode1"] = ext_m1
        external_rows["vlm_lora_mode2"] = ext_m2

    metric_keys = ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "qwk", "mae", "rmse", "parse_rate"]
    drop_rows: List[Dict[str, Any]] = []
    for lane, ext in external_rows.items():
        internal = internal_rows.get(lane, {})
        for mk in metric_keys:
            iv = _safe_float(internal.get(mk))
            ev = _safe_float(ext.get(mk))
            if iv is None or ev is None:
                continue
            drop_rows.append(
                {
                    "lane": lane,
                    "metric": mk,
                    "internal_value": iv,
                    "external_value": ev,
                    "delta_external_minus_internal": float(ev - iv),
                }
            )

    drop_df = pd.DataFrame(drop_rows)
    drop_csv = out_base / "pass7_external_drop_table.csv"
    drop_df.to_csv(drop_csv, index=False)

    report_payload = {
        "task": "pass7_external_validation",
        "generated_utc": _utc_now(),
        "config": {
            "limuc_root": str(limuc_root),
            "meta_csv": str(meta_csv),
            "split": str(args.split),
            "tag": str(args.tag),
            "max_samples": int(args.max_samples),
            "seed": int(args.seed),
            "force_cuda": bool(args.force_cuda),
            "skip_resnet": bool(args.skip_resnet),
            "skip_vlm": bool(args.skip_vlm),
            "resnet_checkpoint": str(args.resnet_checkpoint.resolve()),
            "vlm_run_dir": str(args.vlm_run_dir.resolve()),
            "model_name": str(args.model_name),
            "eval_mode2_strategy": str(args.eval_mode2_strategy),
        },
        "executed_commands": executed_cmds,
        "internal_metrics": internal_rows,
        "external_metrics": external_rows,
        "drop_rows": drop_rows,
        "paths": {
            "out_base": str(out_base),
            "drop_csv": str(drop_csv),
            "resnet_eval_out": str(resnet_eval_out),
            "vlm_run_root": str(vlm_run_root),
            "logs_dir": str(logs_dir),
        },
    }
    report_json = out_base / "pass7_external_validation_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# {args.tag} External Validation Report")
    lines.append("")
    lines.append(f"- Generated UTC: `{_utc_now()}`")
    lines.append(f"- Meta CSV: `{meta_csv}`")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Max samples: `{args.max_samples}`")
    lines.append("")
    lines.append("## Execution")
    lines.append("")
    for row in executed_cmds:
        lines.append(f"- {row['stage']}: `{row['command']}`")
        lines.append(f"  - log: `{row['log']}`")
    if not executed_cmds:
        lines.append("- No command executed (`skip-resnet` and `skip-vlm` both true).")
    lines.append("")
    lines.append("## Internal vs External Delta (external - internal)")
    lines.append("")
    if drop_rows:
        for lane in sorted(set(r["lane"] for r in drop_rows)):
            lines.append(f"- {lane}:")
            lane_rows = [r for r in drop_rows if r["lane"] == lane]
            for r in lane_rows:
                lines.append(
                    f"  - {r['metric']}: internal={r['internal_value']:.4f}, "
                    f"external={r['external_value']:.4f}, delta={r['delta_external_minus_internal']:+.4f}"
                )
    else:
        lines.append("- No comparable metric rows found.")
    lines.append("")

    report_md = out_base / "pass7_external_validation_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print("[pass7] complete")
    print(f"[pass7] report_json={report_json}")
    print(f"[pass7] report_md={report_md}")
    print(f"[pass7] drop_csv={drop_csv}")


if __name__ == "__main__":
    main()
