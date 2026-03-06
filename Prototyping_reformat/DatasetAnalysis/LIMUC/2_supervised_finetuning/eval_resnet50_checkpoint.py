#!/usr/bin/env python3
"""Evaluate a pretrained ResNet50 Mayo classifier on a metadata split."""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


def _ensure_lzma_module() -> None:
    """Provide a lightweight lzma shim when Python is built without _lzma."""

    try:
        import lzma as _lzma  # noqa: F401

        return
    except Exception:
        pass

    mod = types.ModuleType("lzma")

    class LZMAError(Exception):
        pass

    def _unsupported(*_args: Any, **_kwargs: Any) -> Any:
        raise LZMAError("lzma is unavailable in this Python build")

    mod.LZMAError = LZMAError
    mod.open = _unsupported
    mod.compress = _unsupported
    mod.decompress = _unsupported
    mod.LZMAFile = type("LZMAFile", (), {"__init__": _unsupported})
    mod.LZMACompressor = type("LZMACompressor", (), {"__init__": _unsupported})
    mod.LZMADecompressor = type("LZMADecompressor", (), {"__init__": _unsupported})

    for name in [
        "FORMAT_AUTO",
        "FORMAT_XZ",
        "FORMAT_ALONE",
        "FORMAT_RAW",
        "CHECK_NONE",
        "CHECK_CRC32",
        "CHECK_CRC64",
        "CHECK_SHA256",
        "FILTER_LZMA1",
        "FILTER_LZMA2",
        "FILTER_DELTA",
        "FILTER_X86",
        "FILTER_IA64",
        "FILTER_ARM",
        "FILTER_ARMTHUMB",
        "FILTER_POWERPC",
        "FILTER_SPARC",
        "MF_HC3",
        "MF_HC4",
        "MF_BT2",
        "MF_BT3",
        "MF_BT4",
        "MODE_FAST",
        "MODE_NORMAL",
        "PRESET_DEFAULT",
        "PRESET_EXTREME",
    ]:
        setattr(mod, name, 0)

    sys.modules["lzma"] = mod


_ensure_lzma_module()

from torchvision import models, transforms


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--limuc-root",
        type=Path,
        default=None,
        help="Optional LIMUC root used when metadata paths are relative.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--run-name", type=str, default="resnet50_external_eval")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_image_path(meta_csv: Path, raw_path: str, limuc_root: Path | None) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p

    candidates: List[Path] = []
    if limuc_root is not None:
        candidates.append((limuc_root / "0_dataset_prep" / p).resolve())
        candidates.append((limuc_root / p).resolve())
    candidates.append((meta_csv.parent.parent.parent / p).resolve())
    candidates.append((meta_csv.parent / p).resolve())

    for cand in candidates:
        if cand.exists():
            return cand

    return candidates[0] if candidates else p


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray | None, n_bins: int = 10) -> float | None:
    if y_prob is None:
        return None
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    label_names: Sequence[str],
    y_prob: np.ndarray | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    summary: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if y_prob is not None:
        try:
            summary["auroc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            summary["auroc_ovr"] = None
        summary["ece"] = expected_calibration_error(y_true, y_prob, n_bins=10)
    return summary, report


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.paths = frame["image_path"].tolist()
        self.labels = frame["label_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = int(self.labels[idx])
        return x, y


def _load_split_frame(
    *,
    meta_csv: Path,
    split: str,
    limuc_root: Path | None,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    if "split" not in df.columns:
        raise RuntimeError("metadata must include 'split' column")
    if "img_id" not in df.columns:
        df["img_id"] = np.arange(len(df)).astype(str)

    if "label_id" in df.columns and df["label_id"].notna().any():
        df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    else:
        if "label_name" not in df.columns:
            raise RuntimeError("metadata must include label_id or label_name")
        label_names = sorted(str(x) for x in df["label_name"].dropna().unique().tolist())
        name_to_id = {name: i for i, name in enumerate(label_names)}
        df["label_id"] = df["label_name"].map(lambda x: name_to_id.get(str(x)))

    df = df[df["split"] == split].reset_index(drop=True).copy()
    df = df[df["label_id"].notna()].reset_index(drop=True).copy()
    df["label_id"] = df["label_id"].astype(int)
    df["image_path"] = df["image_path"].map(
        lambda p: str(_resolve_image_path(meta_csv=meta_csv, raw_path=str(p), limuc_root=limuc_root))
    )
    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    if max_samples and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No rows left for split='{split}' after filtering/resolve checks.")

    return df


def _run_eval(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    log_every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y.detach().cpu().numpy())
            if log_every > 0 and (batch_idx % log_every == 0):
                print(f"[eval_resnet50] processed_batches={batch_idx}/{len(dataloader)}", flush=True)

    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.asarray([], dtype=int)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.asarray([], dtype=int)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.asarray([], dtype=float)
    return y_true, y_pred, y_prob


def _save_confusion_png(cm: np.ndarray, labels: Sequence[int], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(include_values=False, cmap="Blues", ax=ax, xticks_rotation=90)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    meta_csv = args.meta_csv.resolve()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    limuc_root = args.limuc_root.resolve() if args.limuc_root else None

    if not meta_csv.exists():
        raise FileNotFoundError(f"meta-csv not found: {meta_csv}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = _resolve_device(args.device)
    print(f"[eval_resnet50] meta_csv={meta_csv}")
    print(f"[eval_resnet50] split={args.split}")
    print(f"[eval_resnet50] checkpoint={checkpoint}")
    print(f"[eval_resnet50] out_dir={out_dir}")
    print(f"[eval_resnet50] device={device}")

    frame = _load_split_frame(
        meta_csv=meta_csv,
        split=args.split,
        limuc_root=limuc_root,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"[eval_resnet50] rows={len(frame)}")

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pin_memory = device.type == "cuda"
    dataloader = DataLoader(
        ImageDataset(frame, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 4)
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected checkpoint payload format.")
    model.load_state_dict(payload, strict=True)
    model = model.to(device)

    y_true, y_pred, y_prob = _run_eval(dataloader=dataloader, model=model, device=device, log_every=args.log_every)
    labels = [0, 1, 2, 3]
    label_names = [str(x) for x in labels]
    summary, report = compute_metrics(y_true=y_true, y_pred=y_pred, labels=labels, label_names=label_names, y_prob=y_prob)

    metrics_payload = {"split": args.split, "summary": summary, "report": report}
    metrics_path = out_dir / f"metrics_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    if args.split == "test":
        (out_dir / "metrics_test.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    per_class = {k: v for k, v in report.items() if k in label_names}
    pd.DataFrame(per_class).T.to_csv(out_dir / f"per_class_{args.split}.csv", index=True)
    if args.split == "test":
        pd.DataFrame(per_class).T.to_csv(out_dir / "per_class_test.csv", index=True)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pred_df["img_id"] = frame["img_id"].values
    pred_df["image_path"] = frame["image_path"].values
    for i in labels:
        pred_df[f"prob_{i}"] = y_prob[:, i]
    pred_path = out_dir / f"pred_{args.split}.csv"
    pred_df.to_csv(pred_path, index=False)
    if args.split == "test":
        pred_df.to_csv(out_dir / "pred_test.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_dir / f"confusion_{args.split}.npy", cm)
    _save_confusion_png(
        cm=cm,
        labels=labels,
        out_path=out_dir / f"confusion_{args.split}.png",
        title=f"{args.split.upper()} Confusion Matrix (ResNet50 checkpoint eval)",
    )
    if args.split == "test":
        np.save(out_dir / "confusion_test.npy", cm)
        _save_confusion_png(
            cm=cm,
            labels=labels,
            out_path=out_dir / "confusion_test.png",
            title="TEST Confusion Matrix (ResNet50 checkpoint eval)",
        )

    run_meta = {
        "model": "resnet50_checkpoint_eval",
        "run_name": args.run_name,
        "timestamp_utc": _utc_now(),
        "meta_csv": str(meta_csv),
        "split": str(args.split),
        "rows": int(len(frame)),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "out_dir": str(out_dir),
        "metrics_summary": summary,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("[eval_resnet50] complete")
    print(f"[eval_resnet50] metrics={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
