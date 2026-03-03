#!/usr/bin/env python3
"""Utilities for LIMUC Chapter 4 reporting from persisted results folders."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

DEFAULT_REQUIRED_ARTIFACTS: Sequence[str] = (
    "metrics_test.json",
    "pred_test.csv",
    "pred_val.csv",
    "run_meta.json",
    "per_class_test.csv",
    "confusion_test.png",
)

SUMMARY_METRIC_KEYS: Sequence[str] = (
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
    "parse_rate",
    "n_invalid",
)

RUN_META_KEYS: Sequence[str] = (
    "model",
    "model_name",
    "seed",
    "epochs",
    "lr",
    "weight_decay",
    "best_val_macro_f1",
    "split_hash",
    "run_id",
    "timestamp_utc",
    "prompt",
)


def _sanitize_scalar(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_limuc_root(start: Path | None = None) -> Path:
    """Locate the LIMUC dataset analysis root folder."""
    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (
            (candidate / "0_dataset_prep").exists()
            and (candidate / "1_frozen_encoders").exists()
            and (candidate / "2_supervised_finetuning").exists()
            and (candidate / "3_vlm_severity").exists()
        ):
            return candidate
    raise RuntimeError(
        f"Could not locate LIMUC root from start path: {start_path}"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def list_run_dirs(dataset_root: Path) -> List[Path]:
    """Return run folders under */results/* that contain at least one marker file."""
    marker_files = {
        "metrics_test.json",
        "pred_test.csv",
        "run_meta.json",
        "confusion_test.png",
        "confusion_test.npy",
        "per_class_test.csv",
    }
    run_dirs: List[Path] = []
    for candidate in sorted(dataset_root.glob("**/results/*")):
        if not candidate.is_dir():
            continue
        if any((candidate / marker).exists() for marker in marker_files):
            run_dirs.append(candidate)
    return run_dirs


def build_run_record(
    run_dir: Path,
    dataset_root: Path,
    required_artifacts: Sequence[str] = DEFAULT_REQUIRED_ARTIFACTS,
    expected_test_rows: int = 1686,
) -> Dict[str, Any]:
    rel_dir = run_dir.relative_to(dataset_root)
    rel_parts = rel_dir.parts
    track = rel_parts[0] if rel_parts else ""
    run_name = run_dir.name

    record: Dict[str, Any] = {
        "track": track,
        "run_name": run_name,
        "run_dir": str(rel_dir),
        "expected_test_rows": expected_test_rows,
    }

    missing_artifacts: List[str] = []
    for artifact in required_artifacts:
        has_artifact = (run_dir / artifact).exists()
        col_name = f"has_{artifact.replace('.', '_')}"
        record[col_name] = int(has_artifact)
        if not has_artifact:
            missing_artifacts.append(artifact)

    metrics = _read_json(run_dir / "metrics_test.json")
    summary = metrics.get("summary", {}) if isinstance(metrics.get("summary"), dict) else {}
    for metric_key in SUMMARY_METRIC_KEYS:
        record[metric_key] = _sanitize_scalar(summary.get(metric_key))

    run_meta = _read_json(run_dir / "run_meta.json")
    for key in RUN_META_KEYS:
        record[key] = _sanitize_scalar(run_meta.get(key))

    test_rows = _count_csv_rows(run_dir / "pred_test.csv")
    val_rows = _count_csv_rows(run_dir / "pred_val.csv")
    record["test_rows"] = test_rows
    record["val_rows"] = val_rows
    record["is_full_run"] = int(test_rows == expected_test_rows) if test_rows is not None else 0
    record["is_smoke_or_subset"] = int(
        test_rows is not None and test_rows != expected_test_rows
    )
    record["missing_artifact_count"] = len(missing_artifacts)
    record["missing_artifacts"] = ";".join(missing_artifacts)

    return record


def collect_run_records(
    dataset_root: Path,
    required_artifacts: Sequence[str] = DEFAULT_REQUIRED_ARTIFACTS,
    expected_test_rows: int = 1686,
) -> List[Dict[str, Any]]:
    return [
        build_run_record(
            run_dir=run_dir,
            dataset_root=dataset_root,
            required_artifacts=required_artifacts,
            expected_test_rows=expected_test_rows,
        )
        for run_dir in list_run_dirs(dataset_root)
    ]


def build_missing_artifact_rows(
    run_records: Iterable[Dict[str, Any]],
    required_artifacts: Sequence[str] = DEFAULT_REQUIRED_ARTIFACTS,
) -> List[Dict[str, Any]]:
    missing_rows: List[Dict[str, Any]] = []
    for record in run_records:
        for artifact in required_artifacts:
            col_name = f"has_{artifact.replace('.', '_')}"
            if int(record.get(col_name, 0)) == 0:
                missing_rows.append(
                    {
                        "track": record.get("track"),
                        "run_name": record.get("run_name"),
                        "run_dir": record.get("run_dir"),
                        "missing_artifact": artifact,
                    }
                )
    return missing_rows


def build_main_table_rows(
    run_records: Iterable[Dict[str, Any]],
    full_only: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in run_records:
        if full_only and int(r.get("is_full_run", 0)) != 1:
            continue
        rows.append(
            {
                "run_name": r.get("run_name"),
                "track": r.get("track"),
                "model": r.get("model"),
                "model_name": r.get("model_name"),
                "test_rows": r.get("test_rows"),
                "accuracy": r.get("accuracy"),
                "macro_f1": r.get("macro_f1"),
                "balanced_acc": r.get("balanced_accuracy"),
                "qwk": r.get("qwk"),
                "mae": r.get("mae"),
                "rmse": r.get("rmse"),
                "parse_rate": r.get("parse_rate"),
                "split_hash": r.get("split_hash"),
                "run_dir": r.get("run_dir"),
            }
        )
    rows.sort(
        key=lambda row: (
            -1.0 if row.get("accuracy") is None else -float(row["accuracy"]),
            str(row.get("run_name", "")),
        )
    )
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path, field_order: Sequence[str] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        discovered_fields = list(rows[0].keys())
        for row in rows[1:]:
            for key in row.keys():
                if key not in discovered_fields:
                    discovered_fields.append(key)
        fieldnames = list(field_order) if field_order else discovered_fields
        for key in discovered_fields:
            if key not in fieldnames:
                fieldnames.append(key)
    else:
        fieldnames = list(field_order) if field_order else []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def canonical_image_id(value: Any) -> str:
    """Normalize run-specific image identifiers to a stable comparable key."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    # If a full path is provided, compare by filename stem.
    if "/" in text or "\\" in text:
        text = Path(text).stem

    text = text.lower()
    text = re.sub(r"^mayo_[0-3]__", "", text)
    # Supervised runs often append a running index (e.g., _009590).
    text = re.sub(r"_\d{4,}$", "", text)
    return text


def normalize_prediction_df(pred: pd.DataFrame) -> pd.DataFrame | None:
    """Unify prediction CSV schemas used across supervised and generative runs."""
    df = pred.copy()

    rename_map: Dict[str, str] = {}
    if "img_id" not in df.columns and "image_id" in df.columns:
        rename_map["image_id"] = "img_id"
    if "y_true" not in df.columns and "true_label" in df.columns:
        rename_map["true_label"] = "y_true"
    if "y_pred" not in df.columns and "pred_label" in df.columns:
        rename_map["pred_label"] = "y_pred"
    if "raw_text" not in df.columns and "raw_generation" in df.columns:
        rename_map["raw_generation"] = "raw_text"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"y_true", "y_pred"}
    if not required.issubset(set(df.columns)):
        return None

    if "img_id" not in df.columns:
        if "image_path" in df.columns:
            df["img_id"] = df["image_path"].astype(str).map(lambda p: Path(p).stem)
        else:
            df["img_id"] = [f"row_{i}" for i in range(len(df))]

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    # Prefer stable image-path stem when available.
    if "image_path" in df.columns:
        from_path = df["image_path"].astype(str).map(canonical_image_id)
        from_id = df["img_id"].astype(str).map(canonical_image_id)
        df["img_id_canonical"] = from_path.where(from_path != "", from_id)
    else:
        df["img_id_canonical"] = df["img_id"].astype(str).map(canonical_image_id)

    if "parse_ok" not in df.columns:
        df["parse_ok"] = df["y_pred"].isin([0, 1, 2, 3])

    return df
