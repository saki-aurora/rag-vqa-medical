#!/usr/bin/env python3
"""Deep LIMUC data-quality audit for Pass 1 (repro + integrity)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


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


def _read_split_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    values: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                values.append(s)
    return values


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_split_hash(df: pd.DataFrame) -> str:
    # Match the canonical notebook split-hash definition exactly:
    # hash over sorted "img_id|split" pairs only.
    key = df[["img_id", "split"]].astype(str).sort_values(["img_id", "split"], ascending=[True, True])
    h = hashlib.sha256()
    for row in key.itertuples(index=False):
        h.update(f"{row.img_id}|{row.split}".encode("utf-8"))
    return h.hexdigest()


def _dhash_64_from_image(img: Image.Image) -> int:
    gray = ImageOps.grayscale(img)
    # 9x8 for 64 relative differences.
    small = gray.resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    val = 0
    for bit in bits:
        val = (val << 1) | int(bool(bit))
    return int(val)


def _hamming_distance_64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _laplacian_variance(gray_arr: np.ndarray) -> float:
    # Lightweight Laplacian approximation without OpenCV dependency.
    c = gray_arr[1:-1, 1:-1]
    up = gray_arr[:-2, 1:-1]
    down = gray_arr[2:, 1:-1]
    left = gray_arr[1:-1, :-2]
    right = gray_arr[1:-1, 2:]
    lap = 4.0 * c - up - down - left - right
    return float(np.var(lap))


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if finite.sum() == 0:
        return out
    x = values[finite]
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad <= 1e-12:
        return out
    out[finite] = np.abs((values[finite] - median) / (1.4826 * mad))
    return out


@dataclass
class BKNode:
    value: int
    idxs: List[int]
    children: Dict[int, "BKNode"]


class BKTree64:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def insert(self, value: int, idx: int) -> None:
        if self.root is None:
            self.root = BKNode(value=value, idxs=[idx], children={})
            return
        node = self.root
        while True:
            d = _hamming_distance_64(value, node.value)
            if d == 0:
                node.idxs.append(idx)
                return
            child = node.children.get(d)
            if child is None:
                node.children[d] = BKNode(value=value, idxs=[idx], children={})
                return
            node = child

    def query(self, value: int, threshold: int) -> List[Tuple[int, int]]:
        if self.root is None:
            return []
        out: List[Tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = _hamming_distance_64(value, node.value)
            if d <= threshold:
                for idx in node.idxs:
                    out.append((d, idx))
            lo = d - threshold
            hi = d + threshold
            for edge_dist, child in node.children.items():
                if lo <= edge_dist <= hi:
                    stack.append(child)
        return out


def _iter_image_stats(
    metadata_df: pd.DataFrame,
    progress_every: int = 500,
) -> Iterable[Dict[str, Any]]:
    total = len(metadata_df)
    for i, row in enumerate(metadata_df.itertuples(index=False), start=1):
        img_path = Path(str(row.image_path))
        rec: Dict[str, Any] = {
            "split": row.split,
            "img_id": row.img_id,
            "image_path": str(img_path),
            "label_id": int(row.label_id),
            "label_name": str(row.label_name),
            "patient_id": int(row.patient_id),
            "exists": img_path.exists(),
            "read_ok": False,
            "read_error": None,
            "sha256": None,
            "dhash64": None,
            "width": None,
            "height": None,
            "brightness_mean": None,
            "contrast_std": None,
            "sharpness_laplacian_var": None,
        }
        if not img_path.exists():
            rec["read_error"] = "missing_file"
            yield rec
            continue
        try:
            rec["sha256"] = _sha256_file(img_path)
            with Image.open(img_path) as img:
                rec["dhash64"] = _dhash_64_from_image(img)
                gray = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
                h, w = gray.shape
                rec["width"] = int(w)
                rec["height"] = int(h)
                rec["brightness_mean"] = float(np.mean(gray))
                rec["contrast_std"] = float(np.std(gray))
                rec["sharpness_laplacian_var"] = _laplacian_variance(gray)
            rec["read_ok"] = True
        except Exception as ex:
            rec["read_error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
        if i % progress_every == 0 or i == total:
            print(f"[image-audit] processed {i}/{total}")
        yield rec


def _build_split_consistency_tables(
    metadata_df: pd.DataFrame,
    splits_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    mismatch_rows: List[Dict[str, Any]] = []
    metadata_ids = {
        split: set(metadata_df.loc[metadata_df["split"] == split, "img_id"].astype(str).tolist())
        for split in ["train", "val", "test"]
    }
    for split in ["train", "val", "test"]:
        txt_path = splits_dir / f"{split}.txt"
        file_ids = set(_read_split_txt(txt_path))
        ids_meta = metadata_ids[split]
        missing_in_file = sorted(ids_meta - file_ids)
        extra_in_file = sorted(file_ids - ids_meta)
        rows.append(
            {
                "split": split,
                "split_file": str(txt_path),
                "n_meta_ids": len(ids_meta),
                "n_file_ids": len(file_ids),
                "n_missing_in_file": len(missing_in_file),
                "n_extra_in_file": len(extra_in_file),
                "is_match": int(len(missing_in_file) == 0 and len(extra_in_file) == 0),
            }
        )
        for img_id in missing_in_file[:2000]:
            mismatch_rows.append({"split": split, "mismatch_type": "missing_in_split_file", "img_id": img_id})
        for img_id in extra_in_file[:2000]:
            mismatch_rows.append({"split": split, "mismatch_type": "extra_in_split_file", "img_id": img_id})
    return pd.DataFrame(rows), pd.DataFrame(mismatch_rows)


def _build_patient_leakage_table(metadata_df: pd.DataFrame) -> pd.DataFrame:
    by_patient = (
        metadata_df.groupby("patient_id")
        .agg(
            splits=("split", lambda x: sorted(set(map(str, x)))),
            n_images=("img_id", "count"),
        )
        .reset_index()
    )
    leak_rows: List[Dict[str, Any]] = []
    for row in by_patient.itertuples(index=False):
        splits = list(row.splits)
        if len(splits) <= 1:
            continue
        patient_df = metadata_df[metadata_df["patient_id"] == row.patient_id]
        counts = patient_df.groupby("split").size().to_dict()
        label_counts = (
            patient_df.groupby(["split", "label_id"])
            .size()
            .reset_index(name="count")
            .sort_values(["split", "label_id"])
        )
        label_summary = ";".join(
            f"{r.split}:label{int(r.label_id)}={int(r.count)}"
            for r in label_counts.itertuples(index=False)
        )
        for a, b in combinations(splits, 2):
            leak_rows.append(
                {
                    "patient_id": int(row.patient_id),
                    "split_a": a,
                    "split_b": b,
                    "n_images_split_a": int(counts.get(a, 0)),
                    "n_images_split_b": int(counts.get(b, 0)),
                    "all_splits_for_patient": ",".join(splits),
                    "label_distribution": label_summary,
                }
            )
    if not leak_rows:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "split_a",
                "split_b",
                "n_images_split_a",
                "n_images_split_b",
                "all_splits_for_patient",
                "label_distribution",
            ]
        )
    return pd.DataFrame(leak_rows).sort_values(["split_a", "split_b", "patient_id"]).reset_index(drop=True)


def _build_exact_duplicate_tables(image_audit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = image_audit_df[image_audit_df["sha256"].notna()].copy()
    groups = valid.groupby("sha256", dropna=False)

    group_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []
    for sha, g in groups:
        if len(g) <= 1:
            continue
        splits = sorted(set(g["split"].astype(str).tolist()))
        patients = sorted(set(g["patient_id"].astype(int).tolist()))
        group_rows.append(
            {
                "sha256": sha,
                "group_size": int(len(g)),
                "n_splits": len(splits),
                "splits": ",".join(splits),
                "n_patients": len(patients),
                "cross_split": int(len(splits) > 1),
            }
        )
        for r in g.sort_values(["split", "img_id"]).itertuples(index=False):
            member_rows.append(
                {
                    "sha256": sha,
                    "split": r.split,
                    "img_id": r.img_id,
                    "patient_id": int(r.patient_id),
                    "label_id": int(r.label_id),
                    "image_path": r.image_path,
                }
            )
    group_df = pd.DataFrame(group_rows)
    if not group_df.empty:
        group_df = group_df.sort_values(["cross_split", "group_size"], ascending=[False, False]).reset_index(drop=True)
    member_df = pd.DataFrame(member_rows)
    return group_df, member_df


def _build_near_duplicate_pairs(
    image_audit_df: pd.DataFrame,
    *,
    threshold: int,
    max_pairs: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    usable = image_audit_df[
        (image_audit_df["read_ok"] == True)  # noqa: E712
        & image_audit_df["dhash64"].notna()
    ].copy()
    usable = usable.reset_index(drop=True)
    if usable.empty:
        return pd.DataFrame(), {
            "threshold": threshold,
            "max_pairs": max_pairs,
            "pairs_found": 0,
            "pairs_truncated": False,
            "cross_split_pairs": 0,
            "cross_split_same_patient_pairs": 0,
        }

    tree = BKTree64()
    pair_rows: List[Dict[str, Any]] = []
    truncated = False
    seen_keys: set[Tuple[str, str]] = set()
    total_pairs = 0
    cross_split_pairs = 0
    cross_split_same_patient = 0
    distance_hist: Dict[int, int] = defaultdict(int)
    cross_split_distance_hist: Dict[int, int] = defaultdict(int)

    for i, row in enumerate(usable.itertuples(index=False)):
        hval = int(row.dhash64)
        matches = tree.query(hval, threshold=threshold)
        for dist, j in matches:
            if dist <= 0:
                # dist=0 already captured by exact duplicate hash audit.
                continue
            other = usable.iloc[j]
            key = tuple(sorted((str(other.img_id), str(row.img_id))))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            is_cross_split = int(str(other["split"]) != str(row.split))
            is_same_patient = int(int(other["patient_id"]) == int(row.patient_id))
            total_pairs += 1
            cross_split_pairs += is_cross_split
            if is_cross_split and is_same_patient:
                cross_split_same_patient += 1
            distance_hist[int(dist)] += 1
            if is_cross_split:
                cross_split_distance_hist[int(dist)] += 1
            if len(pair_rows) < max_pairs:
                pair_rows.append(
                    {
                        "distance_hamming": int(dist),
                        "split_a": str(other["split"]),
                        "split_b": str(row.split),
                        "img_id_a": str(other["img_id"]),
                        "img_id_b": str(row.img_id),
                        "patient_id_a": int(other["patient_id"]),
                        "patient_id_b": int(row.patient_id),
                        "label_id_a": int(other["label_id"]),
                        "label_id_b": int(row.label_id),
                        "sha256_a": str(other["sha256"]),
                        "sha256_b": str(row.sha256),
                        "cross_split": is_cross_split,
                        "same_patient": is_same_patient,
                        "image_path_a": str(other["image_path"]),
                        "image_path_b": str(row.image_path),
                    }
                )
            else:
                truncated = True
        tree.insert(hval, i)
        if (i + 1) % 1000 == 0 or (i + 1) == len(usable):
            print(f"[near-duplicate] indexed {i + 1}/{len(usable)} images")

    pair_df = pd.DataFrame(pair_rows)
    return pair_df, {
        "threshold": threshold,
        "max_pairs": max_pairs,
        "pairs_total": int(total_pairs),
        "pairs_stored": int(len(pair_df)),
        "pairs_truncated": bool(truncated),
        "cross_split_pairs": int(cross_split_pairs),
        "cross_split_same_patient_pairs": int(cross_split_same_patient),
        "distance_hist": {str(k): int(v) for k, v in sorted(distance_hist.items())},
        "cross_split_distance_hist": {str(k): int(v) for k, v in sorted(cross_split_distance_hist.items())},
    }


def _build_quality_tables(image_audit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    valid = image_audit_df[image_audit_df["read_ok"] == True].copy()  # noqa: E712
    if valid.empty:
        empty = pd.DataFrame()
        return empty, empty, {"n_valid_images": 0, "outlier_count": 0, "outlier_rate": 0.0}

    # Resolution mode for outlier detection.
    mode_res = (
        valid.groupby(["width", "height"]).size().sort_values(ascending=False).reset_index(name="count").iloc[0]
    )
    mode_w = int(mode_res["width"])
    mode_h = int(mode_res["height"])

    for metric in ["brightness_mean", "contrast_std", "sharpness_laplacian_var"]:
        vals = pd.to_numeric(valid[metric], errors="coerce").to_numpy(dtype=float)
        valid[f"{metric}_rz"] = _robust_zscore(vals)
        valid[f"{metric}_outlier"] = (valid[f"{metric}_rz"] > 3.5).astype(int)

    valid["resolution_outlier"] = ((valid["width"] != mode_w) | (valid["height"] != mode_h)).astype(int)
    valid["any_quality_outlier"] = (
        (valid["brightness_mean_outlier"] == 1)
        | (valid["contrast_std_outlier"] == 1)
        | (valid["sharpness_laplacian_var_outlier"] == 1)
        | (valid["resolution_outlier"] == 1)
    ).astype(int)

    summary_by_split_class = (
        valid.groupby(["split", "label_id", "label_name"])
        .agg(
            n_images=("img_id", "count"),
            brightness_mean=("brightness_mean", "mean"),
            brightness_std=("brightness_mean", "std"),
            contrast_mean=("contrast_std", "mean"),
            contrast_std=("contrast_std", "std"),
            sharpness_mean=("sharpness_laplacian_var", "mean"),
            sharpness_std=("sharpness_laplacian_var", "std"),
            outlier_count=("any_quality_outlier", "sum"),
            outlier_rate=("any_quality_outlier", "mean"),
            width_mode=("width", lambda x: int(pd.Series(x).mode().iloc[0])),
            height_mode=("height", lambda x: int(pd.Series(x).mode().iloc[0])),
        )
        .reset_index()
        .sort_values(["split", "label_id"])
    )

    outlier_rows = valid[valid["any_quality_outlier"] == 1].copy()
    outlier_rows = outlier_rows[
        [
            "split",
            "img_id",
            "patient_id",
            "label_id",
            "label_name",
            "image_path",
            "width",
            "height",
            "brightness_mean",
            "contrast_std",
            "sharpness_laplacian_var",
            "brightness_mean_rz",
            "contrast_std_rz",
            "sharpness_laplacian_var_rz",
            "brightness_mean_outlier",
            "contrast_std_outlier",
            "sharpness_laplacian_var_outlier",
            "resolution_outlier",
            "any_quality_outlier",
        ]
    ].sort_values(
        [
            "resolution_outlier",
            "sharpness_laplacian_var_outlier",
            "contrast_std_outlier",
            "brightness_mean_outlier",
            "split",
            "img_id",
        ],
        ascending=[False, False, False, False, True, True],
    )

    quality_summary = {
        "n_valid_images": int(len(valid)),
        "resolution_mode": {"width": mode_w, "height": mode_h},
        "outlier_count": int(valid["any_quality_outlier"].sum()),
        "outlier_rate": float(valid["any_quality_outlier"].mean()),
        "brightness_mean_global": float(valid["brightness_mean"].mean()),
        "contrast_mean_global": float(valid["contrast_std"].mean()),
        "sharpness_mean_global": float(valid["sharpness_laplacian_var"].mean()),
    }
    return summary_by_split_class, outlier_rows, quality_summary


def _registry_split_hash_health(registry_summary_path: Path) -> Dict[str, Any]:
    payload = _read_json(registry_summary_path)
    if not payload:
        return {"registry_summary_found": False}
    split_summary = payload.get("chapter4_split_hash_summary", {})
    return {
        "registry_summary_found": True,
        "chapter4_split_hash_summary": split_summary,
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
        "--metadata-csv",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/out/metadata/metadata_enriched.csv"),
        help="Path to metadata_enriched.csv.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/out/manifests/image_manifest.csv"),
        help="Path to image_manifest.csv.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/out/splits"),
        help="Path to split txt directory.",
    )
    parser.add_argument(
        "--split-hash-file",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/out/metadata/split_hash.txt"),
        help="Path to canonical split hash file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out"),
        help="Output directory.",
    )
    parser.add_argument(
        "--registry-summary-json",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass1_chapter45_experiment_registry_summary.json"),
        help="Registry summary JSON from 12_build_experiment_registry.py (optional health check).",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=4,
        help="Max Hamming distance for near-duplicate dHash matching.",
    )
    parser.add_argument(
        "--max-near-pairs",
        type=int,
        default=100000,
        help="Safety cap on stored near-duplicate pairs.",
    )
    parser.add_argument(
        "--reuse-image-audit-cache",
        action="store_true",
        help="Reuse existing per-image audit cache if row count matches metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    metadata_csv = args.metadata_csv.resolve()
    manifest_csv = args.manifest_csv.resolve()
    splits_dir = args.splits_dir.resolve()
    split_hash_file = args.split_hash_file.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(metadata_csv)
    required_cols = {"split", "img_id", "image_path", "label_id", "label_name", "patient_id"}
    missing_cols = sorted(required_cols - set(metadata_df.columns))
    if missing_cols:
        raise RuntimeError(f"Missing required metadata columns: {missing_cols}")

    metadata_df = metadata_df.copy()
    metadata_df["split"] = metadata_df["split"].astype(str)
    metadata_df["img_id"] = metadata_df["img_id"].astype(str)
    metadata_df["image_path"] = metadata_df["image_path"].astype(str)
    metadata_df["label_id"] = pd.to_numeric(metadata_df["label_id"], errors="coerce").astype(int)
    metadata_df["patient_id"] = pd.to_numeric(metadata_df["patient_id"], errors="coerce").astype(int)

    split_counts_df = (
        metadata_df.groupby(["split", "label_id", "label_name"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["split", "label_id"])
    )
    split_counts_csv = out_dir / "pass1_split_class_distribution.csv"
    split_counts_df.to_csv(split_counts_csv, index=False)

    patient_counts_df = (
        metadata_df.groupby("split")
        .agg(
            n_images=("img_id", "count"),
            n_patients=("patient_id", "nunique"),
            n_labels=("label_id", "nunique"),
        )
        .reset_index()
        .sort_values("split")
    )
    patient_counts_csv = out_dir / "pass1_split_patient_distribution.csv"
    patient_counts_df.to_csv(patient_counts_csv, index=False)

    split_consistency_df, split_mismatch_df = _build_split_consistency_tables(metadata_df, splits_dir=splits_dir)
    split_consistency_csv = out_dir / "pass1_split_file_consistency.csv"
    split_consistency_df.to_csv(split_consistency_csv, index=False)
    split_mismatch_csv = out_dir / "pass1_split_file_mismatches.csv"
    split_mismatch_df.to_csv(split_mismatch_csv, index=False)

    manifest_exists_health = {"manifest_found": manifest_csv.exists(), "manifest_path": str(manifest_csv)}
    if manifest_csv.exists():
        manifest_df = pd.read_csv(manifest_csv)
        manifest_exists_health["manifest_rows"] = int(len(manifest_df))
        if "exists" in manifest_df.columns:
            manifest_exists_health["manifest_missing_files"] = int((manifest_df["exists"].astype(str) != "True").sum())
        else:
            manifest_exists_health["manifest_missing_files"] = None

    computed_split_hash = _stable_split_hash(metadata_df)
    recorded_split_hash = split_hash_file.read_text(encoding="utf-8").strip() if split_hash_file.exists() else None

    image_audit_cache_csv = out_dir / "pass1_image_audit_rows.csv"
    image_audit_df: pd.DataFrame
    reused_cache = False
    if args.reuse_image_audit_cache and image_audit_cache_csv.exists():
        maybe_cache = pd.read_csv(image_audit_cache_csv)
        if len(maybe_cache) == len(metadata_df):
            reused_cache = True
            image_audit_df = maybe_cache
        else:
            image_audit_df = pd.DataFrame(list(_iter_image_stats(metadata_df)))
            image_audit_df.to_csv(image_audit_cache_csv, index=False)
    else:
        image_audit_df = pd.DataFrame(list(_iter_image_stats(metadata_df)))
        image_audit_df.to_csv(image_audit_cache_csv, index=False)

    patient_leakage_df = _build_patient_leakage_table(metadata_df)
    patient_leakage_csv = out_dir / "pass1_patient_leakage_pairs.csv"
    patient_leakage_df.to_csv(patient_leakage_csv, index=False)

    exact_group_df, exact_member_df = _build_exact_duplicate_tables(image_audit_df)
    exact_group_csv = out_dir / "pass1_exact_duplicate_groups.csv"
    exact_member_csv = out_dir / "pass1_exact_duplicate_members.csv"
    exact_group_df.to_csv(exact_group_csv, index=False)
    exact_member_df.to_csv(exact_member_csv, index=False)

    near_pair_df, near_summary = _build_near_duplicate_pairs(
        image_audit_df,
        threshold=int(args.near_threshold),
        max_pairs=int(args.max_near_pairs),
    )
    near_pair_csv = out_dir / "pass1_near_duplicate_pairs.csv"
    near_pair_df.to_csv(near_pair_csv, index=False)

    quality_summary_df, quality_outlier_df, quality_summary = _build_quality_tables(image_audit_df)
    quality_summary_csv = out_dir / "pass1_image_quality_by_split_class.csv"
    quality_outliers_csv = out_dir / "pass1_image_quality_outliers.csv"
    quality_summary_df.to_csv(quality_summary_csv, index=False)
    quality_outlier_df.to_csv(quality_outliers_csv, index=False)

    read_errors = int((image_audit_df["read_ok"].astype(str) != "True").sum())
    exact_cross_split_groups = int(exact_group_df["cross_split"].sum()) if not exact_group_df.empty else 0
    near_cross_split_pairs = int(near_summary["cross_split_pairs"])

    split_file_pass = bool((split_consistency_df["is_match"] == 1).all())
    split_hash_pass = bool(recorded_split_hash is not None and recorded_split_hash == computed_split_hash)
    patient_leakage_pass = len(patient_leakage_df) == 0
    exact_duplicate_pass = exact_cross_split_groups == 0
    image_read_pass = read_errors == 0

    # Near-duplicates can indicate dataset similarity without being exact leakage.
    near_cross_split_same_patient_pairs = int(near_summary["cross_split_same_patient_pairs"])
    near_duplicate_strict_pass = near_cross_split_pairs == 0
    near_duplicate_patient_leakage_pass = near_cross_split_same_patient_pairs == 0

    status = "PASS"
    if not all(
        [
            split_file_pass,
            split_hash_pass,
            patient_leakage_pass,
            exact_duplicate_pass,
            image_read_pass,
            near_duplicate_patient_leakage_pass,
        ]
    ):
        status = "FAIL"
    elif (
        not near_duplicate_strict_pass
        or bool(near_summary.get("pairs_truncated"))
        or float(quality_summary.get("outlier_rate", 0.0)) > 0.02
    ):
        status = "WARN"

    registry_health = _registry_split_hash_health(args.registry_summary_json.resolve())

    summary_payload = {
        "generated_utc": _utc_now(),
        "status": status,
        "paths": {
            "limuc_root": str(limuc_root),
            "metadata_csv": str(metadata_csv),
            "manifest_csv": str(manifest_csv),
            "splits_dir": str(splits_dir),
            "split_hash_file": str(split_hash_file),
            "image_audit_cache_csv": str(image_audit_cache_csv),
            "split_class_distribution_csv": str(split_counts_csv),
            "split_patient_distribution_csv": str(patient_counts_csv),
            "split_file_consistency_csv": str(split_consistency_csv),
            "split_file_mismatches_csv": str(split_mismatch_csv),
            "patient_leakage_csv": str(patient_leakage_csv),
            "exact_duplicate_groups_csv": str(exact_group_csv),
            "exact_duplicate_members_csv": str(exact_member_csv),
            "near_duplicate_pairs_csv": str(near_pair_csv),
            "image_quality_by_split_class_csv": str(quality_summary_csv),
            "image_quality_outliers_csv": str(quality_outliers_csv),
        },
        "checks": {
            "split_file_consistency_pass": split_file_pass,
            "split_hash_matches_recorded_pass": split_hash_pass,
            "patient_leakage_pass": patient_leakage_pass,
            "exact_duplicate_cross_split_pass": exact_duplicate_pass,
            "near_duplicate_cross_split_pass": near_duplicate_strict_pass,
            "near_duplicate_cross_split_same_patient_pass": near_duplicate_patient_leakage_pass,
            "image_read_pass": image_read_pass,
        },
        "counts": {
            "n_metadata_rows": int(len(metadata_df)),
            "n_unique_img_ids": int(metadata_df["img_id"].nunique()),
            "n_unique_image_paths": int(metadata_df["image_path"].nunique()),
            "n_unique_patients": int(metadata_df["patient_id"].nunique()),
            "n_split_file_mismatches": int(len(split_mismatch_df)),
            "n_patient_leakage_pairs": int(len(patient_leakage_df)),
            "n_exact_duplicate_groups": int(len(exact_group_df)),
            "n_exact_duplicate_cross_split_groups": exact_cross_split_groups,
            "n_near_duplicate_pairs": int(near_summary.get("pairs_total", len(near_pair_df))),
            "n_near_duplicate_pairs_stored": int(near_summary.get("pairs_stored", len(near_pair_df))),
            "n_near_duplicate_cross_split_pairs": near_cross_split_pairs,
            "n_near_duplicate_cross_split_same_patient_pairs": near_cross_split_same_patient_pairs,
            "n_image_read_errors": read_errors,
            "n_quality_outliers": int(quality_summary.get("outlier_count", 0)),
            "quality_outlier_rate": float(quality_summary.get("outlier_rate", 0.0)),
        },
        "split_hash": {
            "recorded_split_hash": recorded_split_hash,
            "computed_split_hash": computed_split_hash,
        },
        "manifest_health": manifest_exists_health,
        "near_duplicate_summary": near_summary,
        "quality_summary": quality_summary,
        "registry_health": registry_health,
        "image_audit_cache_reused": reused_cache,
    }

    summary_json = out_dir / "pass1_data_quality_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {split_counts_csv}")
    print(f"Wrote: {patient_counts_csv}")
    print(f"Wrote: {split_consistency_csv}")
    print(f"Wrote: {split_mismatch_csv}")
    print(f"Wrote: {image_audit_cache_csv}")
    print(f"Wrote: {patient_leakage_csv}")
    print(f"Wrote: {exact_group_csv}")
    print(f"Wrote: {exact_member_csv}")
    print(f"Wrote: {near_pair_csv}")
    print(f"Wrote: {quality_summary_csv}")
    print(f"Wrote: {quality_outliers_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
