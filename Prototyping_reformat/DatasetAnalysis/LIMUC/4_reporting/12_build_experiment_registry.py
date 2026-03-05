#!/usr/bin/env python3
"""Build a unified Chapter 4 + Chapter 5 experiment registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ALLOWED_LIMUC_TRACKS = {"1_frozen_encoders", "2_supervised_finetuning", "3_vlm_severity"}
CHAPTER4_FULL_TEST_ROWS = 1686


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
    except Exception:
        return None


def _count_jsonl_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return None
    return count


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _row_base(
    *,
    experiment_group: str,
    entity_type: str,
    run_name: str,
    run_id: str | None,
    timestamp_utc: str | None,
    artifact_dir: str,
) -> Dict[str, Any]:
    return {
        "experiment_group": experiment_group,
        "entity_type": entity_type,
        "run_name": run_name,
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "artifact_dir": artifact_dir,
        "track": None,
        "lane": None,
        "run_scope": None,
        "model": None,
        "model_name": None,
        "seed": None,
        "split_hash": None,
        "n_train_rows": None,
        "n_val_rows": None,
        "n_test_rows": None,
        "n_queries": None,
        "n_outputs": None,
        "accuracy": None,
        "macro_f1": None,
        "balanced_acc": None,
        "qwk": None,
        "mae": None,
        "rmse": None,
        "parse_rate": None,
        "p_at_1": None,
        "r_at_1": None,
        "hit_at_1": None,
        "p_at_3": None,
        "r_at_3": None,
        "hit_at_3": None,
        "p_at_5": None,
        "r_at_5": None,
        "hit_at_5": None,
        "citation_coverage": None,
        "citation_correctness_heuristic": None,
        "hallucination_rate_proxy": None,
        "refusal_count": None,
        "kb_docs": None,
        "kb_chunks": None,
        "kb_source_files": None,
        "index_backend": None,
        "status": None,
        "run_meta_path": None,
        "metrics_path": None,
        "pred_test_path": None,
        "config_path": None,
        "outputs_path": None,
        "notes": None,
    }


def _classify_chapter4_lane(track: str, run_name: str, model: str | None) -> str:
    if track == "3_vlm_severity":
        if run_name.startswith("single_image_eval"):
            return "generative_diagnostic"
        return "generative"
    return "supervised"


def _classify_run_scope(n_test_rows: int | None, expected_test_rows: int) -> str:
    if n_test_rows is None:
        return "unknown"
    if n_test_rows == expected_test_rows:
        return "full"
    if n_test_rows <= 64:
        return "smoke"
    return "subset"


def collect_chapter4_registry_rows(
    limuc_root: Path,
    expected_test_rows: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(limuc_root.glob("**/results/*")):
        if not run_dir.is_dir():
            continue
        rel = run_dir.relative_to(limuc_root)
        if not rel.parts or rel.parts[0] not in ALLOWED_LIMUC_TRACKS:
            continue

        track = rel.parts[0]
        run_name = run_dir.name
        run_meta_path = run_dir / "run_meta.json"
        metrics_path = run_dir / "metrics_test.json"
        pred_test_path = run_dir / "pred_test.csv"

        run_meta = _read_json(run_meta_path)
        metrics = _read_json(metrics_path)
        summary = metrics.get("summary", {}) if isinstance(metrics.get("summary"), dict) else {}

        n_test_rows = _count_csv_rows(pred_test_path)
        n_val_rows = _count_csv_rows(run_dir / "pred_val.csv")
        n_train_rows = _count_csv_rows(run_dir / "pred_train.csv")

        model = str(run_meta.get("model")) if run_meta.get("model") is not None else None
        row = _row_base(
            experiment_group="chapter4",
            entity_type="model_run",
            run_name=run_name,
            run_id=run_meta.get("run_id"),
            timestamp_utc=run_meta.get("timestamp_utc"),
            artifact_dir=str(run_dir.resolve()),
        )
        row.update(
            {
                "track": track,
                "lane": _classify_chapter4_lane(track, run_name, model),
                "run_scope": _classify_run_scope(n_test_rows, expected_test_rows),
                "model": model,
                "model_name": run_meta.get("model_name"),
                "seed": _safe_int(run_meta.get("seed")),
                "split_hash": run_meta.get("split_hash"),
                "n_train_rows": n_train_rows,
                "n_val_rows": n_val_rows,
                "n_test_rows": n_test_rows,
                "accuracy": _safe_float(summary.get("accuracy")),
                "macro_f1": _safe_float(summary.get("macro_f1")),
                "balanced_acc": _safe_float(summary.get("balanced_accuracy", summary.get("balanced_acc"))),
                "qwk": _safe_float(summary.get("qwk")),
                "mae": _safe_float(summary.get("mae")),
                "rmse": _safe_float(summary.get("rmse")),
                "parse_rate": _safe_float(summary.get("parse_rate")),
                "run_meta_path": str(run_meta_path.resolve()) if run_meta_path.exists() else None,
                "metrics_path": str(metrics_path.resolve()) if metrics_path.exists() else None,
                "pred_test_path": str(pred_test_path.resolve()) if pred_test_path.exists() else None,
            }
        )
        rows.append(row)
    return rows


def collect_chapter5_registry_rows(chapter5_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    results_root = chapter5_root / "results"
    if not results_root.exists():
        return rows

    for manifest_path in sorted(results_root.glob("*/kb_manifest.json")):
        manifest = _read_json(manifest_path)
        run_name = manifest_path.parent.name
        row = _row_base(
            experiment_group="chapter5",
            entity_type="kb_build",
            run_name=run_name,
            run_id=f"kb_build_{run_name}",
            timestamp_utc=manifest.get("created_utc"),
            artifact_dir=str(manifest_path.parent.resolve()),
        )
        row.update(
            {
                "kb_docs": _safe_int(manifest.get("n_docs")),
                "kb_chunks": _safe_int(manifest.get("n_chunks")),
                "kb_source_files": _safe_int(manifest.get("n_source_files")),
                "index_backend": manifest.get("index_backend"),
                "config_path": str(manifest_path.resolve()),
                "notes": f"index_file={manifest.get('index_file')}",
            }
        )
        rows.append(row)

    for config_path in sorted(results_root.glob("*/run_config.json")):
        cfg = _read_json(config_path)
        run_dir = config_path.parent
        run_name = run_dir.name
        run_id = cfg.get("run_id") if cfg.get("run_id") else f"wrapper_{run_name}"
        outputs_path = run_dir / "wrapper_outputs.jsonl"
        row = _row_base(
            experiment_group="chapter5",
            entity_type="wrapper_run",
            run_name=run_name,
            run_id=str(run_id),
            timestamp_utc=None,
            artifact_dir=str(run_dir.resolve()),
        )
        row.update(
            {
                "run_scope": "full" if _safe_int(cfg.get("n_queries")) and _safe_int(cfg.get("n_queries")) >= 20 else "manual_or_smoke",
                "n_queries": _safe_int(cfg.get("n_queries")),
                "n_outputs": _count_jsonl_rows(outputs_path),
                "config_path": str(config_path.resolve()),
                "outputs_path": str(outputs_path.resolve()) if outputs_path.exists() else None,
                "notes": (
                    f"mode={cfg.get('mode_requested')};retrieval_k={cfg.get('retrieval_k')};"
                    f"manifest={cfg.get('manifest_path')};severity_context={cfg.get('has_severity_context')}"
                ),
            }
        )
        rows.append(row)

    for pico_eval_path in sorted(results_root.glob("*/pico_eval.json")):
        payload = _read_json(pico_eval_path)
        run_name = pico_eval_path.parent.name
        row = _row_base(
            experiment_group="chapter5",
            entity_type="eval_pico",
            run_name=run_name,
            run_id=f"eval_pico_{run_name}",
            timestamp_utc=None,
            artifact_dir=str(pico_eval_path.parent.resolve()),
        )
        row.update(
            {
                "macro_f1": _safe_float(payload.get("macro_f1_all_fields")),
                "notes": (
                    f"macro_f1_required_fields={payload.get('macro_f1_required_fields')};"
                    f"n_queries={payload.get('n_queries')}"
                ),
                "metrics_path": str(pico_eval_path.resolve()),
            }
        )
        rows.append(row)

    for retrieval_eval_path in sorted(results_root.glob("*/retrieval_eval.json")):
        payload = _read_json(retrieval_eval_path)
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        k1 = metrics.get("k=1", {}) if isinstance(metrics.get("k=1"), dict) else {}
        k3 = metrics.get("k=3", {}) if isinstance(metrics.get("k=3"), dict) else {}
        k5 = metrics.get("k=5", {}) if isinstance(metrics.get("k=5"), dict) else {}
        run_name = retrieval_eval_path.parent.name
        row = _row_base(
            experiment_group="chapter5",
            entity_type="eval_retrieval",
            run_name=run_name,
            run_id=f"eval_retrieval_{run_name}",
            timestamp_utc=None,
            artifact_dir=str(retrieval_eval_path.parent.resolve()),
        )
        row.update(
            {
                "p_at_1": _safe_float(k1.get("precision_at_k")),
                "r_at_1": _safe_float(k1.get("recall_at_k")),
                "hit_at_1": _safe_float(k1.get("hit_rate_at_k")),
                "p_at_3": _safe_float(k3.get("precision_at_k")),
                "r_at_3": _safe_float(k3.get("recall_at_k")),
                "hit_at_3": _safe_float(k3.get("hit_rate_at_k")),
                "p_at_5": _safe_float(k5.get("precision_at_k")),
                "r_at_5": _safe_float(k5.get("recall_at_k")),
                "hit_at_5": _safe_float(k5.get("hit_rate_at_k")),
                "n_queries": _safe_int(payload.get("n_queries")),
                "metrics_path": str(retrieval_eval_path.resolve()),
            }
        )
        rows.append(row)

    for answer_eval_path in sorted(results_root.glob("*/answer_eval.json")):
        payload = _read_json(answer_eval_path)
        run_name = answer_eval_path.parent.name
        row = _row_base(
            experiment_group="chapter5",
            entity_type="eval_answers",
            run_name=run_name,
            run_id=f"eval_answers_{run_name}",
            timestamp_utc=None,
            artifact_dir=str(answer_eval_path.parent.resolve()),
        )
        row.update(
            {
                "n_outputs": _safe_int(payload.get("n_outputs")),
                "citation_coverage": _safe_float(payload.get("citation_coverage")),
                "citation_correctness_heuristic": _safe_float(payload.get("citation_correctness_heuristic")),
                "hallucination_rate_proxy": _safe_float(payload.get("hallucination_rate_proxy")),
                "refusal_count": _safe_int(payload.get("refusal_count")),
                "metrics_path": str(answer_eval_path.resolve()),
            }
        )
        rows.append(row)

    for report_path in sorted(results_root.glob("chapter5_completion_audit_*/chapter5_completion_report.json")):
        payload = _read_json(report_path)
        run_name = report_path.parent.name
        row = _row_base(
            experiment_group="chapter5",
            entity_type="completion_audit",
            run_name=run_name,
            run_id=payload.get("run_id"),
            timestamp_utc=payload.get("generated_utc"),
            artifact_dir=str(report_path.parent.resolve()),
        )
        row.update(
            {
                "status": payload.get("status"),
                "config_path": str(report_path.resolve()),
                "notes": f"report_path={payload.get('report_path')}",
            }
        )
        rows.append(row)

    return rows


def _chapter4_split_hash_summary(df: pd.DataFrame) -> Dict[str, Any]:
    chapter4_full = df[
        (df["experiment_group"] == "chapter4")
        & (df["entity_type"] == "model_run")
        & (df["run_scope"] == "full")
    ].copy()
    hash_counts: Dict[str, int] = {}
    if not chapter4_full.empty:
        vc = chapter4_full["split_hash"].dropna().astype(str).value_counts()
        hash_counts = {k: int(v) for k, v in vc.to_dict().items()}
    return {
        "n_full_runs": int(len(chapter4_full)),
        "split_hash_counts": hash_counts,
        "is_consistent": len(hash_counts) <= 1 and len(chapter4_full) > 0,
    }


def _status_summary(df: pd.DataFrame) -> Dict[str, Any]:
    by_group = df.groupby("experiment_group").size().to_dict()
    by_entity = df.groupby("entity_type").size().to_dict()
    return {
        "total_records": int(len(df)),
        "records_by_group": {str(k): int(v) for k, v in by_group.items()},
        "records_by_entity_type": {str(k): int(v) for k, v in by_entity.items()},
    }


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
        "--expected-test-rows",
        type=int,
        default=CHAPTER4_FULL_TEST_ROWS,
        help="Expected full test row count for Chapter 4 runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    chapter5_root = args.chapter5_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chapter4_rows = collect_chapter4_registry_rows(limuc_root=limuc_root, expected_test_rows=args.expected_test_rows)
    chapter5_rows = collect_chapter5_registry_rows(chapter5_root=chapter5_root)
    all_rows = chapter4_rows + chapter5_rows
    if not all_rows:
        raise RuntimeError("No records discovered for Chapter 4/5 registry.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        by=["experiment_group", "entity_type", "timestamp_utc", "run_name"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    registry_csv = out_dir / "pass1_chapter45_experiment_registry.csv"
    df.to_csv(registry_csv, index=False)

    chapter4_full_csv = out_dir / "pass1_chapter4_full_runs_registry.csv"
    chapter4_full_df = df[
        (df["experiment_group"] == "chapter4")
        & (df["entity_type"] == "model_run")
        & (df["run_scope"] == "full")
    ].copy()
    chapter4_full_df = chapter4_full_df.sort_values(by=["accuracy", "run_name"], ascending=[False, True])
    chapter4_full_df.to_csv(chapter4_full_csv, index=False)

    split_hash_summary = _chapter4_split_hash_summary(df)
    summary_payload = {
        "generated_utc": _utc_now(),
        "limuc_root": str(limuc_root),
        "chapter5_root": str(chapter5_root),
        "registry_csv": str(registry_csv),
        "chapter4_full_runs_csv": str(chapter4_full_csv),
        **_status_summary(df),
        "chapter4_split_hash_summary": split_hash_summary,
    }
    summary_json = out_dir / "pass1_chapter45_experiment_registry_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote registry CSV: {registry_csv}")
    print(f"Wrote Chapter 4 full-runs CSV: {chapter4_full_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Total records: {summary_payload['total_records']}")


if __name__ == "__main__":
    main()
