#!/usr/bin/env python3
"""One-command Pass 1 runner: experiment registry + data-quality audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run(cmd: List[str]) -> None:
    print(f"[pass1] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _bool_icon(v: bool) -> str:
    return "PASS" if v else "FAIL"


def _build_report_md(
    *,
    registry_summary: Dict[str, Any],
    data_quality_summary: Dict[str, Any],
    registry_summary_json_path: Path,
    data_quality_summary_json_path: Path,
    out_path: Path,
) -> None:
    checks = data_quality_summary.get("checks", {})
    counts = data_quality_summary.get("counts", {})
    status = data_quality_summary.get("status", "UNKNOWN")
    registry_split = registry_summary.get("chapter4_split_hash_summary", {})

    lines: List[str] = []
    lines.append(f"# Pass 1 Repro + Data-Quality Audit ({status})")
    lines.append("")
    lines.append(f"- generated_utc: `{_utc_now()}`")
    lines.append("")
    lines.append("## 1) Checklist")
    lines.append("")
    lines.append(f"- {_bool_icon(bool(checks.get('split_file_consistency_pass')))} split files consistent with metadata")
    lines.append(f"- {_bool_icon(bool(checks.get('split_hash_matches_recorded_pass')))} computed split hash matches recorded hash")
    lines.append(f"- {_bool_icon(bool(checks.get('patient_leakage_pass')))} no patient leakage across splits")
    lines.append(f"- {_bool_icon(bool(checks.get('exact_duplicate_cross_split_pass')))} no exact cross-split duplicates")
    lines.append(f"- {_bool_icon(bool(checks.get('near_duplicate_cross_split_pass')))} no near-duplicate cross-split pairs (dHash)")
    lines.append(
        f"- {_bool_icon(bool(checks.get('near_duplicate_cross_split_same_patient_pass')))} "
        "no near-duplicate cross-split pairs involving same patient"
    )
    lines.append(f"- {_bool_icon(bool(checks.get('image_read_pass')))} no image read/decode failures")
    lines.append("")
    lines.append("## 2) Core Counts")
    lines.append("")
    lines.append(f"- metadata rows: `{counts.get('n_metadata_rows')}`")
    lines.append(f"- unique patients: `{counts.get('n_unique_patients')}`")
    lines.append(f"- split file mismatches: `{counts.get('n_split_file_mismatches')}`")
    lines.append(f"- patient leakage pairs: `{counts.get('n_patient_leakage_pairs')}`")
    lines.append(f"- exact duplicate groups (cross-split): `{counts.get('n_exact_duplicate_cross_split_groups')}`")
    lines.append(f"- near duplicate cross-split pairs: `{counts.get('n_near_duplicate_cross_split_pairs')}`")
    lines.append(f"- near duplicate cross-split same-patient pairs: `{counts.get('n_near_duplicate_cross_split_same_patient_pairs')}`")
    lines.append(f"- image read errors: `{counts.get('n_image_read_errors')}`")
    lines.append(f"- quality outlier rate: `{counts.get('quality_outlier_rate')}`")
    lines.append("")
    lines.append("## 3) Registry Health")
    lines.append("")
    lines.append(f"- total registry records: `{registry_summary.get('total_records')}`")
    lines.append(f"- Chapter 4 full runs in registry: `{registry_split.get('n_full_runs')}`")
    lines.append(f"- Chapter 4 split-hash consistent in registry: `{registry_split.get('is_consistent')}`")
    lines.append(f"- Chapter 4 split hash counts: `{registry_split.get('split_hash_counts')}`")
    lines.append("")
    lines.append("## 4) Evidence Files")
    lines.append("")
    paths = data_quality_summary.get("paths", {})
    for key in sorted(paths.keys()):
        lines.append(f"- `{key}`: `{paths[key]}`")
    lines.append("")
    lines.append(f"- `registry_csv`: `{registry_summary.get('registry_csv')}`")
    lines.append(f"- `registry_summary_json`: `{registry_summary_json_path}`")
    lines.append(f"- `data_quality_summary_json`: `{data_quality_summary_json_path}`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limuc-root",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC"),
        help="Path to LIMUC root.",
    )
    parser.add_argument(
        "--chapter5-root",
        type=Path,
        default=Path("Prototyping_reformat/chapter5_pico_wrapper"),
        help="Path to Chapter 5 wrapper root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out"),
        help="Output directory.",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=4,
        help="Near-duplicate dHash threshold.",
    )
    parser.add_argument(
        "--max-near-pairs",
        type=int,
        default=100000,
        help="Safety cap for near-duplicate pair rows.",
    )
    parser.add_argument(
        "--reuse-image-audit-cache",
        action="store_true",
        help="Reuse pass1 image audit cache when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    chapter5_root = args.chapter5_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    this_dir = Path(__file__).resolve().parent
    registry_script = this_dir / "12_build_experiment_registry.py"
    data_quality_script = this_dir / "13_data_quality_audit.py"
    if not registry_script.exists() or not data_quality_script.exists():
        raise RuntimeError("Expected pass1 scripts 12/13 are missing.")

    _run(
        [
            sys.executable,
            str(registry_script),
            "--limuc-root",
            str(limuc_root),
            "--chapter5-root",
            str(chapter5_root),
            "--out-dir",
            str(out_dir),
        ]
    )

    dq_cmd = [
        sys.executable,
        str(data_quality_script),
        "--limuc-root",
        str(limuc_root),
        "--out-dir",
        str(out_dir),
        "--near-threshold",
        str(args.near_threshold),
        "--max-near-pairs",
        str(args.max_near_pairs),
        "--registry-summary-json",
        str(out_dir / "pass1_chapter45_experiment_registry_summary.json"),
    ]
    if args.reuse_image_audit_cache:
        dq_cmd.append("--reuse-image-audit-cache")
    _run(dq_cmd)

    registry_summary_path = out_dir / "pass1_chapter45_experiment_registry_summary.json"
    dq_summary_path = out_dir / "pass1_data_quality_summary.json"
    registry_summary = _read_json(registry_summary_path)
    dq_summary = _read_json(dq_summary_path)
    if not registry_summary or not dq_summary:
        raise RuntimeError("Pass 1 summary artifacts missing after run.")

    consolidated = {
        "run_id": f"pass1_repro_data_quality_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_utc": _utc_now(),
        "status": dq_summary.get("status", "UNKNOWN"),
        "registry_summary_json": str(registry_summary_path),
        "data_quality_summary_json": str(dq_summary_path),
        "registry_summary": registry_summary,
        "data_quality_summary": dq_summary,
    }
    consolidated_json = out_dir / "pass1_repro_data_quality_report.json"
    consolidated_json.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")

    consolidated_md = out_dir / "pass1_repro_data_quality_report.md"
    _build_report_md(
        registry_summary=registry_summary,
        data_quality_summary=dq_summary,
        registry_summary_json_path=registry_summary_path,
        data_quality_summary_json_path=dq_summary_path,
        out_path=consolidated_md,
    )

    print(f"Wrote: {consolidated_json}")
    print(f"Wrote: {consolidated_md}")
    print(f"Status: {consolidated.get('status')}")


if __name__ == "__main__":
    main()
