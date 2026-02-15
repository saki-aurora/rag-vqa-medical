from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from scipy import ndimage as ndi


KVASIR_SEG_ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_SESSILE_ZIP_URL = "https://datasets.simula.no/downloads/kvasir-sessile.zip"


# -----------------------------
# Path + I/O helpers
# -----------------------------
def find_kvasir_seg_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if p.name == "Kvasir_SEG" and (p / "0_dataset_prep").exists():
            return p
    raise RuntimeError(f"Could not locate Kvasir_SEG root from cwd={start}")


def _has_image_mask_tree(path: Path) -> bool:
    return (path / "images").exists() and (path / "masks").exists()


def _find_first_kvasir_seg_tree(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if _has_image_mask_tree(path):
        return path
    for p in path.rglob("*"):
        if p.is_dir() and _has_image_mask_tree(p):
            return p
    return None


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    found = _find_first_kvasir_seg_tree(out_dir)
    if found is None:
        raise FileNotFoundError(f"Could not find extracted Kvasir-SEG tree in {out_dir}")
    return found


def ensure_kvasir_seg_data(
    root: Path,
    allow_download: bool = False,
    seg_zip_url: str = KVASIR_SEG_ZIP_URL,
) -> Path:
    """Return directory that contains images/ and masks/ for Kvasir-SEG."""
    candidates = [
        root / "0_dataset_prep" / "Kvasir-SEG",
        root / "0_dataset_prep" / "out" / "Kvasir-SEG",
        root / "0_dataset_prep" / "out" / "raw" / "Kvasir-SEG",
    ]
    for c in candidates:
        if _has_image_mask_tree(c):
            return c

    zip_candidates = [
        root / "0_dataset_prep" / "out" / "kvasir-seg.zip",
        root / "0_dataset_prep" / "kvasir-seg.zip",
    ]
    for z in zip_candidates:
        if z.exists():
            extracted = _extract_zip(z, root / "0_dataset_prep" / "out" / "raw")
            return extracted

    if allow_download:
        z = root / "0_dataset_prep" / "out" / "kvasir-seg.zip"
        _download_file(seg_zip_url, z)
        extracted = _extract_zip(z, root / "0_dataset_prep" / "out" / "raw")
        return extracted

    raise FileNotFoundError(
        "Kvasir-SEG data not found. Expected extracted folder under 0_dataset_prep/ "
        "or zip at 0_dataset_prep/out/kvasir-seg.zip"
    )


def ensure_kvasir_sessile_zip(root: Path, allow_download: bool = False) -> Optional[Path]:
    candidates = [
        root / "0_dataset_prep" / "out" / "kvasir-sessile.zip",
        root / "0_dataset_prep" / "kvasir-sessile.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    if not allow_download:
        return None
    z = root / "0_dataset_prep" / "out" / "kvasir-sessile.zip"
    _download_file(KVASIR_SESSILE_ZIP_URL, z)
    return z


def _find_first_image_mask_tree(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if _has_image_mask_tree(path):
        return path
    for p in path.rglob("*"):
        if p.is_dir() and _has_image_mask_tree(p):
            return p
    return None


def _find_first_sessile_tree(path: Path) -> Optional[Path]:
    if not path.exists():
        return None

    candidates = []
    if _has_image_mask_tree(path):
        candidates.append(path)
    for p in path.rglob("*"):
        if p.is_dir() and _has_image_mask_tree(p):
            candidates.append(p)

    if not candidates:
        return None

    sessile_candidates = [p for p in candidates if "sessile" in str(p).lower()]
    if not sessile_candidates:
        return None

    ranked = sorted(sessile_candidates, key=lambda p: len(p.parts))
    return ranked[0]


def ensure_kvasir_sessile_data(
    root: Path,
    allow_download: bool = False,
    sessile_zip_url: str = KVASIR_SESSILE_ZIP_URL,
) -> Path:
    """Return directory that contains images/ and masks/ for Kvasir-Sessile."""
    candidates = [
        root / "0_dataset_prep" / "out" / "raw" / "kvasir_sessile",
        root / "0_dataset_prep" / "out" / "raw" / "sessile-main-Kvasir-SEG",
        root / "0_dataset_prep" / "out" / "raw",
    ]
    for c in candidates:
        found = _find_first_sessile_tree(c)
        if found is not None:
            return found

    zip_path = ensure_kvasir_sessile_zip(root, allow_download=allow_download)
    if zip_path is None:
        raise FileNotFoundError(
            "Kvasir-Sessile zip not found. Expected 0_dataset_prep/out/kvasir-sessile.zip "
            "or enable allow_download=True."
        )

    extract_root = root / "0_dataset_prep" / "out" / "raw" / "kvasir_sessile"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    found = _find_first_sessile_tree(extract_root)
    if found is None:
        raise FileNotFoundError(f"Could not find extracted Kvasir-Sessile tree in {extract_root}")
    return found


def load_bboxes_json(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


# -----------------------------
# Metadata + split helpers
# -----------------------------
def _binary_mask_from_path(mask_path: Path, threshold: int = 127) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    return (arr > threshold).astype(np.uint8)


def _compute_component_count(mask_bin: np.ndarray) -> int:
    if mask_bin.sum() == 0:
        return 0
    _, n = ndi.label(mask_bin)
    return int(n)


def _bbox_count_from_item(item: dict) -> int:
    if not item:
        return 0
    b = item.get("bbox", [])
    if isinstance(b, list):
        return len(b)
    return 0


def _compute_split_hash(df: pd.DataFrame, id_col: str = "img_id", split_col: str = "split") -> str:
    h = hashlib.sha256()
    key_df = df[[id_col, split_col]].astype(str).sort_values([id_col, split_col])
    for row in key_df.itertuples(index=False):
        h.update(f"{row[0]}|{row[1]}".encode("utf-8"))
    return h.hexdigest()


def build_kvasir_seg_metadata(
    data_dir: Path,
    bboxes: Optional[Dict[str, dict]] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    threshold: int = 127,
) -> pd.DataFrame:
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing images/masks dirs under: {data_dir}")

    rows = []
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    for img_path in image_paths:
        stem = img_path.stem
        mask_candidates = [
            masks_dir / f"{stem}.jpg",
            masks_dir / f"{stem}.jpeg",
            masks_dir / f"{stem}.png",
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is None:
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        mask_bin = _binary_mask_from_path(mask_path, threshold=threshold)
        fg_pixels = int(mask_bin.sum())
        total_pixels = int(mask_bin.size)
        fg_ratio = float(fg_pixels / max(total_pixels, 1))
        components = _compute_component_count(mask_bin)

        bbox_info = (bboxes or {}).get(stem, {})
        bbox_count = _bbox_count_from_item(bbox_info)

        rows.append(
            {
                "img_id": stem,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "width": int(width),
                "height": int(height),
                "fg_pixels": fg_pixels,
                "total_pixels": total_pixels,
                "mask_area_ratio": fg_ratio,
                "component_count": components,
                "bbox_count": bbox_count,
            }
        )

    if not rows:
        raise ValueError(f"No image-mask pairs found in {data_dir}")

    df = pd.DataFrame(rows).sort_values("img_id").reset_index(drop=True)

    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    split = np.array(["test"] * n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train : n_train + n_val]] = "val"

    df["split"] = split
    df["split_seed"] = int(seed)
    return df


def build_kvasir_sessile_metadata(
    data_dir: Path,
    threshold: int = 127,
    split_name: str = "external_test",
) -> pd.DataFrame:
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing images/masks dirs under: {data_dir}")

    rows = []
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    for img_path in image_paths:
        stem = img_path.stem
        mask_candidates = [
            masks_dir / f"{stem}.jpg",
            masks_dir / f"{stem}.jpeg",
            masks_dir / f"{stem}.png",
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is None:
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        mask_bin = _binary_mask_from_path(mask_path, threshold=threshold)
        fg_pixels = int(mask_bin.sum())
        total_pixels = int(mask_bin.size)
        fg_ratio = float(fg_pixels / max(total_pixels, 1))
        components = _compute_component_count(mask_bin)

        rows.append(
            {
                "img_id": stem,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "width": int(width),
                "height": int(height),
                "fg_pixels": fg_pixels,
                "total_pixels": total_pixels,
                "mask_area_ratio": fg_ratio,
                "component_count": components,
                "bbox_count": 0,
                "split": str(split_name),
                "split_seed": -1,
                "source_dataset": "kvasir_sessile",
            }
        )

    if not rows:
        raise ValueError(f"No image-mask pairs found in {data_dir}")
    return pd.DataFrame(rows).sort_values("img_id").reset_index(drop=True)


def write_metadata_artifacts(root: Path, df: pd.DataFrame) -> Dict[str, Path]:
    out_root = root / "0_dataset_prep" / "out"
    metadata_dir = out_root / "metadata"
    manifests_dir = out_root / "manifests"
    splits_dir = out_root / "splits"
    for d in [metadata_dir, manifests_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Raw + enriched
    raw_csv = metadata_dir / "metadata_raw.csv"
    enriched_csv = metadata_dir / "metadata_enriched.csv"
    manifest_csv = manifests_dir / "image_mask_manifest.csv"
    split_hash_txt = metadata_dir / "split_hash.txt"

    df.to_csv(raw_csv, index=False)
    df.to_csv(enriched_csv, index=False)

    manifest = df[["img_id", "image_path", "mask_path", "split"]].copy()
    manifest["exists_img"] = manifest["image_path"].map(lambda p: Path(p).exists())
    manifest["exists_mask"] = manifest["mask_path"].map(lambda p: Path(p).exists())
    manifest.to_csv(manifest_csv, index=False)

    for split_name in ["train", "val", "test"]:
        ids = df.loc[df["split"] == split_name, "img_id"].astype(str).sort_values().tolist()
        (splits_dir / f"{split_name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))

    split_hash = _compute_split_hash(df)
    split_hash_txt.write_text(split_hash + "\n")

    return {
        "raw_csv": raw_csv,
        "enriched_csv": enriched_csv,
        "manifest_csv": manifest_csv,
        "split_hash_txt": split_hash_txt,
        "splits_dir": splits_dir,
    }


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"img_id", "image_path", "mask_path", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    return df


# -----------------------------
# Segmentation metrics
# -----------------------------
def _safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / (b + eps))


def binary_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    compute_hd95: bool = False,
) -> Dict[str, float]:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, np.logical_not(true)).sum())
    fn = float(np.logical_and(np.logical_not(pred), true).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(true)).sum())

    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)

    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_area_ratio": float(true.mean()),
        "pred_area_ratio": float(pred.mean()),
    }

    if compute_hd95:
        out["hd95"] = float(compute_hd95_distance(pred, true))
    return out


def compute_hd95_distance(pred: np.ndarray, true: np.ndarray) -> float:
    pred = pred.astype(bool)
    true = true.astype(bool)

    if pred.sum() == 0 and true.sum() == 0:
        return 0.0
    if pred.sum() == 0 or true.sum() == 0:
        return float("nan")

    pred_border = np.logical_xor(pred, ndi.binary_erosion(pred))
    true_border = np.logical_xor(true, ndi.binary_erosion(true))

    if pred_border.sum() == 0 or true_border.sum() == 0:
        return float("nan")

    dt_true = ndi.distance_transform_edt(~true_border)
    dt_pred = ndi.distance_transform_edt(~pred_border)

    d_pred_to_true = dt_true[pred_border]
    d_true_to_pred = dt_pred[true_border]
    if d_pred_to_true.size == 0 or d_true_to_pred.size == 0:
        return float("nan")

    dists = np.concatenate([d_pred_to_true, d_true_to_pred])
    return float(np.percentile(dists, 95.0))


def summarize_metric_frame(df: pd.DataFrame, metrics: Optional[Iterable[str]] = None) -> Dict[str, float]:
    metrics = list(metrics or ["dice", "iou", "precision", "recall", "f1", "specificity"])
    out = {"n": int(len(df))}
    for m in metrics:
        if m in df.columns:
            out[f"{m}_mean"] = float(df[m].mean())
            out[f"{m}_median"] = float(df[m].median())
            out[f"{m}_std"] = float(df[m].std(ddof=0))
    return out


def compute_slice_metrics(per_image_df: pd.DataFrame) -> pd.DataFrame:
    if per_image_df.empty:
        return pd.DataFrame()

    q = per_image_df["true_area_ratio"].quantile([0.25, 0.75]).to_dict()
    q1 = float(q.get(0.25, 0.0))
    q3 = float(q.get(0.75, 0.0))

    def size_bin(v: float) -> str:
        if v <= q1:
            return "small"
        if v >= q3:
            return "large"
        return "medium"

    tmp = per_image_df.copy()
    tmp["size_bin"] = tmp["true_area_ratio"].map(size_bin)
    tmp["component_bin"] = np.where(tmp.get("component_count", 1) > 1, "multi_component", "single_component")

    rows = []
    for by_col in ["size_bin", "component_bin"]:
        for name, g in tmp.groupby(by_col):
            row = {"slice_family": by_col, "slice_name": str(name), "n": int(len(g))}
            for m in ["dice", "iou", "precision", "recall", "f1", "specificity"]:
                if m in g.columns:
                    row[f"{m}_mean"] = float(g[m].mean())
            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Dataset + models
# -----------------------------
class KvasirSegDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 352,
        train: bool = False,
        normalize: bool = True,
        threshold: int = 127,
        seed: int = 42,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        color_jitter_prob: float = 0.0,
        color_jitter_strength: float = 0.2,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_size = int(image_size)
        self.train = bool(train)
        self.normalize = bool(normalize)
        self.threshold = int(threshold)
        self.rng = random.Random(seed)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter_strength = float(color_jitter_strength)
        self.color_jitter = None
        if self.color_jitter_prob > 0.0:
            s = self.color_jitter_strength
            self.color_jitter = transforms.ColorJitter(
                brightness=s,
                contrast=s,
                saturation=s,
                hue=min(0.1, s / 2.0),
            )

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Deterministic random branch per sample call.
        if self.rng.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.rng.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if self.color_jitter is not None and self.rng.random() < self.color_jitter_prob:
            image = self.color_jitter(image)
        return image, mask

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        msk = Image.open(row["mask_path"]).convert("L")

        img = TF.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        if self.train:
            img, msk = self._augment(img, msk)

        img_t = TF.to_tensor(img)
        if self.normalize:
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        msk_np = (np.asarray(msk, dtype=np.uint8) > self.threshold).astype(np.float32)
        msk_t = torch.from_numpy(msk_np).unsqueeze(0)

        return {
            "image": img_t,
            "mask": msk_t,
            "img_id": str(row["img_id"]),
            "image_path": str(row["image_path"]),
            "split": str(row["split"]),
            "component_count": int(row.get("component_count", 1)),
            "true_area_ratio": float(row.get("mask_area_ratio", float(msk_np.mean()))),
        }


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


def build_deeplab_resnet50(num_classes: int = 1) -> nn.Module:
    model = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


class SegformerBinaryWrapper(nn.Module):
    """
    Binary SegFormer wrapper compatible with the training notebook checkpoint format.
    """

    def __init__(
        self,
        config_kwargs: Optional[Dict[str, Any]] = None,
        allow_hf_download: bool = False,
        try_local_pretrained: bool = False,
    ) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        self.source = "config_init"
        model = None

        if try_local_pretrained:
            try:
                model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                )
                self.source = "cached_pretrained_b2"
            except Exception:
                model = None

        if model is None and allow_hf_download:
            try:
                model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                )
                self.source = "downloaded_pretrained_b2"
            except Exception:
                model = None

        if model is None:
            cfg = SegformerConfig(num_labels=1, **(config_kwargs or {}))
            model = SegformerForSemanticSegmentation(cfg)
            self.source = "config_init"

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError("Checkpoint did not contain a state_dict dictionary")

    state_dict: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            state_dict[str(k)] = v

    if not state_dict:
        raise ValueError("Checkpoint state_dict is empty")

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def _state_dict_prefix(state_dict: Dict[str, torch.Tensor], probe: str) -> Optional[str]:
    if any(k.startswith(probe) for k in state_dict.keys()):
        return ""
    if any(k.startswith(f"model.{probe}") for k in state_dict.keys()):
        return "model."
    return None


def infer_segformer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    prefix = _state_dict_prefix(state_dict, "segformer.")
    if prefix is None:
        raise ValueError("State dict does not look like a SegFormer checkpoint")

    hidden_sizes: List[int] = []
    for i in range(4):
        key = f"{prefix}segformer.encoder.patch_embeddings.{i}.proj.weight"
        w = state_dict.get(key)
        if w is None or w.ndim < 1:
            raise ValueError(f"Missing key for SegFormer config inference: {key}")
        hidden_sizes.append(int(w.shape[0]))

    depth_map: Dict[int, int] = {}
    block_prefix = f"{prefix}segformer.encoder.block."
    for k in state_dict.keys():
        if not k.startswith(block_prefix):
            continue
        tail = k[len(block_prefix) :]
        parts = tail.split(".")
        if len(parts) < 2:
            continue
        try:
            stage_idx = int(parts[0])
            block_idx = int(parts[1])
        except ValueError:
            continue
        depth_map[stage_idx] = max(depth_map.get(stage_idx, -1), block_idx)

    depths: List[int] = []
    for stage_idx in range(4):
        depths.append(int(depth_map.get(stage_idx, 1) + 1))

    decoder_key = f"{prefix}decode_head.linear_fuse.weight"
    decoder_hidden_size = 128
    if decoder_key in state_dict and state_dict[decoder_key].ndim >= 1:
        decoder_hidden_size = int(state_dict[decoder_key].shape[0])

    return {
        "hidden_sizes": hidden_sizes,
        "depths": depths,
        "decoder_hidden_size": decoder_hidden_size,
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
    }


def _build_smp_unet_resnet34() -> nn.Module:
    import segmentation_models_pytorch as smp  # type: ignore

    return smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1, in_channels=3)


def _state_dict_looks_like(state_dict: Dict[str, torch.Tensor], pattern: str) -> bool:
    return any(pattern in k for k in state_dict.keys())


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name_hint: Optional[str] = None,
    allow_hf_download: bool = False,
) -> nn.Module:
    state_obj = _torch_load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(state_obj)
    lower_hint = (model_name_hint or "").lower()

    candidates: List[Tuple[str, Any]] = []

    def add_candidate(name: str, builder: Any) -> None:
        if name not in [n for n, _ in candidates]:
            candidates.append((name, builder))

    if "segformer" in lower_hint:
        add_candidate(
            "segformer_inferred",
            lambda: SegformerBinaryWrapper(
                config_kwargs=infer_segformer_config_from_state_dict(state_dict),
                allow_hf_download=allow_hf_download,
                try_local_pretrained=False,
            ),
        )
    if "deeplab" in lower_hint:
        add_candidate("deeplabv3_resnet50", lambda: build_deeplab_resnet50(num_classes=1))
    if "unet_resnet34_smp" in lower_hint or ("unet" in lower_hint and "resnet34" in lower_hint):
        add_candidate("smp_unet_resnet34", _build_smp_unet_resnet34)
        add_candidate("unet_small", lambda: UNetSmall(in_ch=3, out_ch=1, base=32))
    elif "unet" in lower_hint:
        add_candidate("unet_small", lambda: UNetSmall(in_ch=3, out_ch=1, base=32))

    if not candidates:
        if _state_dict_looks_like(state_dict, "segformer.encoder.patch_embeddings"):
            add_candidate(
                "segformer_inferred",
                lambda: SegformerBinaryWrapper(
                    config_kwargs=infer_segformer_config_from_state_dict(state_dict),
                    allow_hf_download=allow_hf_download,
                    try_local_pretrained=False,
                ),
            )
        if _state_dict_looks_like(state_dict, "backbone.") or _state_dict_looks_like(state_dict, "classifier."):
            add_candidate("deeplabv3_resnet50", lambda: build_deeplab_resnet50(num_classes=1))
        if _state_dict_looks_like(state_dict, "enc1.") or _state_dict_looks_like(state_dict, "bottleneck."):
            add_candidate("unet_small", lambda: UNetSmall(in_ch=3, out_ch=1, base=32))
        if _state_dict_looks_like(state_dict, "encoder.") and _state_dict_looks_like(state_dict, "decoder."):
            add_candidate("smp_unet_resnet34", _build_smp_unet_resnet34)

    # Safety net order.
    add_candidate("unet_small", lambda: UNetSmall(in_ch=3, out_ch=1, base=32))
    add_candidate("deeplabv3_resnet50", lambda: build_deeplab_resnet50(num_classes=1))

    errors = []
    for name, builder in candidates:
        try:
            model = builder()
        except Exception as e:
            errors.append(f"{name}: init_failed={e}")
            continue
        try:
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception as e:
            errors.append(f"{name}: load_failed={e}")
            continue

    joined = "\n".join(errors[:8])
    raise RuntimeError(f"Could not load checkpoint {checkpoint_path} with available model builders.\n{joined}")


def extract_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    if hasattr(out, "logits"):
        return out.logits
    return out


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_from_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dloss


def focal_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - pt) ** gamma) * bce
    return loss.mean()


def compute_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    loss_name = str(getattr(cfg, "loss_name", "bce_dice")).lower()

    if loss_name == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if loss_name == "dice":
        return dice_loss_from_logits(logits, targets)
    if loss_name == "focal":
        return focal_loss_from_logits(
            logits,
            targets,
            alpha=float(getattr(cfg, "focal_alpha", 0.25)),
            gamma=float(getattr(cfg, "focal_gamma", 2.0)),
        )
    if loss_name == "focal_dice":
        focal = focal_loss_from_logits(
            logits,
            targets,
            alpha=float(getattr(cfg, "focal_alpha", 0.25)),
            gamma=float(getattr(cfg, "focal_gamma", 2.0)),
        )
        dloss = dice_loss_from_logits(logits, targets)
        return 0.5 * focal + 0.5 * dloss

    # Default = bce_dice
    return bce_dice_loss(logits, targets, bce_weight=float(getattr(cfg, "bce_weight", 0.5)))


# -----------------------------
# Training/evaluation helpers
# -----------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 352
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    threshold: float = 0.5
    compute_hd95: bool = False
    save_pred_masks: bool = True
    pred_mask_limit: int = 200
    loss_name: str = "bce_dice"
    bce_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    hflip_prob: float = 0.5
    vflip_prob: float = 0.2
    color_jitter_prob: float = 0.0
    color_jitter_strength: float = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split_dataloaders(
    df: pd.DataFrame,
    cfg: TrainConfig,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Dict[str, DataLoader]:
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    if max_train:
        train_df = train_df.sample(n=min(max_train, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    if max_val:
        val_df = val_df.sample(n=min(max_val, len(val_df)), random_state=cfg.seed).reset_index(drop=True)
    if max_test:
        test_df = test_df.sample(n=min(max_test, len(test_df)), random_state=cfg.seed).reset_index(drop=True)

    train_ds = KvasirSegDataset(
        train_df,
        image_size=cfg.image_size,
        train=True,
        seed=cfg.seed,
        hflip_prob=cfg.hflip_prob,
        vflip_prob=cfg.vflip_prob,
        color_jitter_prob=cfg.color_jitter_prob,
        color_jitter_strength=cfg.color_jitter_strength,
    )
    val_ds = KvasirSegDataset(
        val_df,
        image_size=cfg.image_size,
        train=False,
        seed=cfg.seed,
        hflip_prob=0.0,
        vflip_prob=0.0,
        color_jitter_prob=0.0,
    )
    test_ds = KvasirSegDataset(
        test_df,
        image_size=cfg.image_size,
        train=False,
        seed=cfg.seed,
        hflip_prob=0.0,
        vflip_prob=0.0,
        color_jitter_prob=0.0,
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }
    return loaders


def _to_device(batch: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    return images, masks


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for batch in loader:
        images, masks = _to_device(batch, device)
        logits = extract_logits(model, images)
        loss = compute_segmentation_loss(logits, masks, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running += float(loss.item()) * bs
        n += bs
    return float(running / max(n, 1))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    compute_hd95: bool = False,
    collect_pred_masks: bool = False,
    loss_cfg: Optional[TrainConfig] = None,
) -> Tuple[float, pd.DataFrame, Dict[str, np.ndarray]]:
    model.eval()
    running_loss = 0.0
    n = 0
    rows = []
    pred_masks: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in loader:
            images, masks = _to_device(batch, device)
            logits = extract_logits(model, images)
            if loss_cfg is not None:
                loss = compute_segmentation_loss(logits, masks, loss_cfg)
            else:
                loss = bce_dice_loss(logits, masks)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            bs = images.size(0)
            running_loss += float(loss.item()) * bs
            n += bs

            preds_np = preds.detach().cpu().numpy()[:, 0]
            masks_np = masks.detach().cpu().numpy()[:, 0]

            for i in range(bs):
                img_id = str(batch["img_id"][i])
                metrics = binary_segmentation_metrics(preds_np[i], masks_np[i], compute_hd95=compute_hd95)
                rows.append(
                    {
                        "img_id": img_id,
                        "split": str(batch["split"][i]),
                        "component_count": int(batch["component_count"][i]),
                        **metrics,
                    }
                )
                if collect_pred_masks:
                    pred_masks[img_id] = (preds_np[i] > 0.5).astype(np.uint8)

    out_df = pd.DataFrame(rows)
    avg_loss = float(running_loss / max(n, 1))
    return avg_loss, out_df, pred_masks


def make_eval_dataloader(
    metadata_df: pd.DataFrame,
    image_size: int = 352,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    split_name: str = "external_test",
) -> DataLoader:
    df = metadata_df.copy().reset_index(drop=True)
    if "split" not in df.columns:
        df["split"] = str(split_name)

    ds = KvasirSegDataset(
        df=df,
        image_size=image_size,
        train=False,
        normalize=True,
        threshold=127,
        seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate_model_on_metadata(
    model: nn.Module,
    metadata_df: pd.DataFrame,
    image_size: int = 352,
    batch_size: int = 4,
    num_workers: int = 2,
    threshold: float = 0.5,
    seed: int = 42,
    compute_hd95: bool = False,
    split_name: str = "external_test",
    device: Optional[torch.device] = None,
    loss_cfg: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = make_eval_dataloader(
        metadata_df=metadata_df,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        split_name=split_name,
    )
    loss, frame, _ = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
        compute_hd95=compute_hd95,
        collect_pred_masks=False,
        loss_cfg=loss_cfg,
    )
    summary = summarize_metric_frame(frame)
    summary["loss"] = float(loss)
    return {
        "summary": summary,
        "per_image": frame,
        "device": str(device),
    }


def read_run_config(run_dir: Path) -> Dict[str, Any]:
    run_cfg_path = run_dir / "run_config.json"
    if not run_cfg_path.exists():
        return {}
    with open(run_cfg_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def evaluate_run_dir_on_metadata(
    run_dir: Path,
    metadata_df: pd.DataFrame,
    out_dir: Path,
    threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_size: Optional[int] = None,
    seed: Optional[int] = None,
    split_name: str = "external_test",
    compute_hd95: bool = False,
    allow_hf_download: bool = False,
) -> Dict[str, Any]:
    run_cfg = read_run_config(run_dir)
    model_name = str(run_cfg.get("model", run_dir.name))
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name_hint=model_name,
        allow_hf_download=allow_hf_download,
    )

    eval_cfg = {
        "image_size": int(image_size if image_size is not None else run_cfg.get("image_size", 352)),
        "batch_size": int(batch_size if batch_size is not None else run_cfg.get("batch_size", 4)),
        "num_workers": int(num_workers if num_workers is not None else run_cfg.get("num_workers", 2)),
        "threshold": float(threshold if threshold is not None else run_cfg.get("threshold", 0.5)),
        "seed": int(seed if seed is not None else run_cfg.get("seed", 42)),
        "split_name": str(split_name),
        "compute_hd95": bool(compute_hd95),
    }
    loss_cfg = TrainConfig(
        seed=eval_cfg["seed"],
        image_size=eval_cfg["image_size"],
        batch_size=eval_cfg["batch_size"],
        num_workers=eval_cfg["num_workers"],
        threshold=eval_cfg["threshold"],
        compute_hd95=eval_cfg["compute_hd95"],
        loss_name=str(run_cfg.get("loss_name", "bce_dice")),
        bce_weight=float(run_cfg.get("bce_weight", 0.5)),
        focal_gamma=float(run_cfg.get("focal_gamma", 2.0)),
        focal_alpha=float(run_cfg.get("focal_alpha", 0.25)),
    )

    res = evaluate_model_on_metadata(
        model=model,
        metadata_df=metadata_df,
        image_size=eval_cfg["image_size"],
        batch_size=eval_cfg["batch_size"],
        num_workers=eval_cfg["num_workers"],
        threshold=eval_cfg["threshold"],
        seed=eval_cfg["seed"],
        compute_hd95=eval_cfg["compute_hd95"],
        split_name=eval_cfg["split_name"],
        loss_cfg=loss_cfg,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / f"per_image_{run_dir.name}.csv"
    metrics_path = out_dir / f"metrics_{run_dir.name}.json"
    res["per_image"].to_csv(per_image_path, index=False)

    payload = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "model_name": model_name,
        "eval_config": eval_cfg,
        "summary": res["summary"],
        "device": res["device"],
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    row = {
        "run_name": run_dir.name,
        "model_name": model_name,
        "per_image_path": str(per_image_path),
        "metrics_path": str(metrics_path),
    }
    for k, v in res["summary"].items():
        row[k] = v
    return row


def save_pred_masks(pred_masks: Dict[str, np.ndarray], out_dir: Path, limit: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img_id, arr) in enumerate(pred_masks.items()):
        if i >= limit:
            break
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img.save(out_dir / f"{img_id}.png")


def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    metadata_df: pd.DataFrame,
    split_hash: Optional[str],
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Dict[str, dict]:
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loaders = make_split_dataloaders(
        metadata_df,
        cfg,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_path = out_dir / "best_model.pt"
    history = []
    best_val_dice = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, cfg)
        val_loss, val_df, _ = evaluate_model(
            model,
            loaders["val"],
            device,
            threshold=cfg.threshold,
            compute_hd95=cfg.compute_hd95,
            collect_pred_masks=False,
            loss_cfg=cfg,
        )
        val_summary = summarize_metric_frame(val_df)
        val_dice = float(val_summary.get("dice_mean", 0.0))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice_mean": val_dice,
                "val_iou_mean": float(val_summary.get("iou_mean", 0.0)),
            }
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

    # Best checkpoint eval
    model.load_state_dict(torch.load(best_path, map_location=device))

    eval_results = {}
    for split in ["train", "val", "test"]:
        collect_masks = split == "test" and cfg.save_pred_masks
        loss, frame, pred_masks = evaluate_model(
            model,
            loaders[split],
            device,
            threshold=cfg.threshold,
            compute_hd95=cfg.compute_hd95,
            collect_pred_masks=collect_masks,
            loss_cfg=cfg,
        )

        frame_path = out_dir / f"per_image_{split}.csv"
        frame.to_csv(frame_path, index=False)

        summary = summarize_metric_frame(frame)
        summary["loss"] = float(loss)

        metrics_path = out_dir / f"metrics_{split}.json"
        with open(metrics_path, "w") as f:
            json.dump({"split": split, "summary": summary}, f, indent=2)

        if split == "test":
            slice_df = compute_slice_metrics(frame)
            slice_df.to_csv(out_dir / "slice_metrics_test.csv", index=False)
            if cfg.save_pred_masks:
                save_pred_masks(pred_masks, out_dir / "pred_masks_test", limit=cfg.pred_mask_limit)

        eval_results[split] = summary

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    run_config = {
        "model": model_name,
        "seed": cfg.seed,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "threshold": cfg.threshold,
        "compute_hd95": cfg.compute_hd95,
        "loss_name": cfg.loss_name,
        "bce_weight": cfg.bce_weight,
        "focal_gamma": cfg.focal_gamma,
        "focal_alpha": cfg.focal_alpha,
        "hflip_prob": cfg.hflip_prob,
        "vflip_prob": cfg.vflip_prob,
        "color_jitter_prob": cfg.color_jitter_prob,
        "color_jitter_strength": cfg.color_jitter_strength,
        "max_train": max_train,
        "max_val": max_val,
        "max_test": max_test,
        "split_hash": split_hash,
        "device": str(device),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    return eval_results


# -----------------------------
# Visualization helpers
# -----------------------------
def save_overlay_grid(
    df: pd.DataFrame,
    out_path: Path,
    n: int = 16,
    seed: int = 42,
) -> None:
    import matplotlib.pyplot as plt

    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    cols = 4
    rows = int(math.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flatten():
        ax.axis("off")

    for i, row in sample.iterrows():
        r, c = divmod(i, cols)
        ax = axes[r, c]

        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        img_np = np.asarray(img)
        m = (np.asarray(mask) > 127)

        overlay = img_np.copy()
        overlay[m, 0] = np.clip(0.6 * overlay[m, 0] + 100, 0, 255)
        overlay[m, 1] = np.clip(0.6 * overlay[m, 1], 0, 255)
        overlay[m, 2] = np.clip(0.6 * overlay[m, 2], 0, 255)

        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"{row['img_id']} ({row['split']})")
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["mask_area_ratio"].values, bins=30)
    ax.set_title("Mask Foreground Ratio")
    ax.set_xlabel("foreground ratio")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "mask_area_ratio_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["component_count"].values, bins=min(20, int(df["component_count"].max()) + 1))
    ax.set_title("Connected Components per Mask")
    ax.set_xlabel("component_count")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "component_count_hist.png", dpi=180)
    plt.close(fig)
