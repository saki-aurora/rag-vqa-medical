#!/usr/bin/env python3
"""Train/evaluate a supervised torchvision backbone for LIMUC Mayo classification."""

from __future__ import annotations

import argparse
import json
import math
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
import torch.optim as optim
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

try:
    from scipy.stats import spearmanr

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limuc-root", type=Path, default=None, help="Path to LIMUC root.")
    parser.add_argument("--meta-csv", type=Path, default=None)
    parser.add_argument("--label-map-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=0)

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "convnext_tiny", "convnext_small", "vit_b_16", "swin_t"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--aug-strength", type=str, default="medium", choices=["light", "medium", "strong"])

    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weighting", type=str, default="balanced", choices=["balanced", "none"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
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


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        t_logp = logp.gather(1, targets.view(-1, 1)).squeeze(1)
        t_p = p.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = -((1.0 - t_p) ** self.gamma) * t_logp
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


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
    if _HAS_SCIPY:
        summary["spearman"] = float(spearmanr(y_true, y_pred).correlation)
    if y_prob is not None:
        try:
            summary["auroc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            summary["auroc_ovr"] = None
        summary["ece"] = expected_calibration_error(y_true, y_prob, n_bins=10)
    return summary, report


def save_split_outputs(
    *,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    label_names: Sequence[str],
    out_dir: Path,
    y_prob: np.ndarray | None,
    df_meta: pd.DataFrame | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary, report = compute_metrics(y_true, y_pred, labels, label_names, y_prob)
    metrics = {"split": split_name, "summary": summary, "report": report}
    (out_dir / f"metrics_{split_name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_class = {k: v for k, v in report.items() if k in label_names}
    pd.DataFrame(per_class).T.to_csv(out_dir / f"per_class_{split_name}.csv", index=True)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if df_meta is not None:
        pred_df["img_id"] = df_meta["img_id"].values
        pred_df["image_path"] = df_meta["image_path"].values
    if y_prob is not None:
        for i in range(y_prob.shape[1]):
            pred_df[f"prob_{i}"] = y_prob[:, i]
    pred_df.to_csv(out_dir / f"pred_{split_name}.csv", index=False)
    return summary, report


def _read_split_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def _build_transforms(args: argparse.Namespace) -> Tuple[transforms.Compose, transforms.Compose]:
    train_aug: List[Any] = [transforms.Resize(args.resize_size)]
    if args.aug_strength == "light":
        train_aug.extend(
            [
                transforms.RandomResizedCrop(args.image_size, scale=(0.92, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    elif args.aug_strength == "medium":
        train_aug.extend(
            [
                transforms.RandomResizedCrop(args.image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(6),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            ]
        )
    else:
        train_aug.extend(
            [
                transforms.RandomResizedCrop(args.image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.08),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            ]
        )

    train_aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(train_aug), eval_tf


def _build_model(backbone: str, n_classes: int) -> nn.Module:
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)
        return model
    if backbone == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)
        return model
    if backbone == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
        return model
    if backbone == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, n_classes)
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


def run_epoch(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    log_every: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch_idx, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if amp_enabled and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                if amp_enabled and device.type == "cuda" and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(y.detach().cpu().numpy())
        total_loss += float(loss.item()) * int(y.size(0))

        if log_every > 0 and (batch_idx % log_every == 0):
            print(f"  batch={batch_idx}/{len(dataloader)} loss={float(loss.item()):.5f}", flush=True)

    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.asarray([], dtype=int)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.asarray([], dtype=int)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else None
    avg_loss = total_loss / max(1, len(dataloader.dataset))
    return avg_loss, y_true, y_pred, y_prob


def main() -> None:
    args = parse_args()
    limuc_root = (args.limuc_root.resolve() if args.limuc_root else find_limuc_root())
    meta_csv = (
        args.meta_csv.resolve()
        if args.meta_csv
        else limuc_root / "0_dataset_prep" / "out" / "metadata" / "metadata_enriched.csv"
    )
    label_map_csv = (
        args.label_map_csv.resolve()
        if args.label_map_csv
        else limuc_root / "0_dataset_prep" / "out" / "metadata" / "label_map.csv"
    )
    if not meta_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {meta_csv}")

    default_name = f"finetune_{args.backbone}_seed{args.seed:03d}"
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (limuc_root / "2_supervised_finetuning" / "results" / default_name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    seed_all(args.seed)

    print(f"[train_supervised] limuc_root={limuc_root}")
    print(f"[train_supervised] out_dir={out_dir}")
    print(f"[train_supervised] device={device}")
    print(
        f"[train_supervised] backbone={args.backbone} loss={args.loss} seed={args.seed} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} weight_decay={args.weight_decay} "
        f"aug={args.aug_strength} image_size={args.image_size} amp={args.amp}"
    )

    meta = pd.read_csv(meta_csv)
    images_base = limuc_root / "0_dataset_prep"

    def _to_abs(p: str) -> str:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((images_base / pp).resolve())

    meta["image_path"] = meta["image_path"].map(_to_abs)
    meta = meta[meta["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    if args.max_samples and args.max_samples > 0:
        meta = meta.sample(n=min(args.max_samples, len(meta)), random_state=args.seed).reset_index(drop=True)

    if label_map_csv.exists():
        label_map = pd.read_csv(label_map_csv)
        id_to_name = {int(r.label_id): str(r.label_name) for _, r in label_map.iterrows()}
    else:
        names = sorted(str(x) for x in meta["label_name"].dropna().unique().tolist())
        id_to_name = {i: name for i, name in enumerate(names)}

    if "label_id" in meta.columns and meta["label_id"].notna().any():
        meta["label_id"] = pd.to_numeric(meta["label_id"], errors="coerce")
    else:
        name_to_id: Dict[Any, int] = {}
        for k, v in id_to_name.items():
            name_to_id[v] = int(k)
            name_to_id[str(v)] = int(k)
        meta["label_id"] = meta["label_name"].map(lambda x: name_to_id.get(x, name_to_id.get(str(x))))

    before_label_filter = len(meta)
    meta = meta[meta["label_id"].notna()].copy()
    meta["label_id"] = meta["label_id"].astype(int)
    removed = before_label_filter - len(meta)
    if removed > 0:
        print(f"[train_supervised][warn] dropped {removed} rows with invalid labels")

    train_df = meta[meta["split"] == "train"].reset_index(drop=True)
    val_df = meta[meta["split"].isin(["val", "validation"])].reset_index(drop=True)
    test_df = meta[meta["split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Unexpected split sizes train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )
    print(f"[train_supervised] split sizes train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_tf, eval_tf = _build_transforms(args)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        ImageDataset(train_df, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ImageDataset(val_df, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        ImageDataset(test_df, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    class_counts = train_df["label_id"].value_counts().sort_index()
    n_classes = int(class_counts.shape[0])
    if n_classes != 4:
        print(f"[train_supervised][warn] expected 4 classes, found {n_classes}")

    class_weights_tensor: torch.Tensor | None = None
    if args.class_weighting == "balanced":
        total = float(class_counts.sum())
        weights = total / (n_classes * class_counts)
        class_weights_tensor = torch.tensor(weights.values, dtype=torch.float32, device=device)

    model = _build_model(args.backbone, n_classes=n_classes).to(device)

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=float(args.label_smoothing))
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=float(args.focal_gamma), weight=class_weights_tensor)
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    scheduler: Any = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)

    labels = sorted(id_to_name.keys())
    label_names = [str(id_to_name[i]) for i in labels]
    best_val_macro_f1 = -1.0
    history: List[Dict[str, float]] = []
    best_model_path = out_dir / "best_model.pt"
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        print(f"[train_supervised] epoch={epoch}/{args.epochs}")
        train_loss, _, _, _ = run_epoch(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=args.amp,
            scaler=scaler,
            log_every=args.log_every,
        )
        val_loss, y_val, p_val, prob_val = run_epoch(
            dataloader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            amp_enabled=args.amp,
            scaler=None,
            log_every=0,
        )
        val_summary, _ = compute_metrics(y_val, p_val, labels, label_names, prob_val)
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "lr": lr_now,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_summary["macro_f1"]),
                "val_qwk": float(val_summary["qwk"]),
            }
        )
        print(
            f"[train_supervised] epoch={epoch} lr={lr_now:.7f} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_macro_f1={float(val_summary['macro_f1']):.5f} "
            f"val_qwk={float(val_summary['qwk']):.5f}"
        )

        if float(val_summary["macro_f1"]) > best_val_macro_f1:
            best_val_macro_f1 = float(val_summary["macro_f1"])
            torch.save(model.state_dict(), best_model_path)
            no_improve_epochs = 0
            print(f"[train_supervised] new best model saved: val_macro_f1={best_val_macro_f1:.5f}")
        else:
            no_improve_epochs += 1

        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"[train_supervised] early stop at epoch={epoch} (patience={args.early_stop_patience})")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, y_train, p_train, prob_train = run_epoch(
        dataloader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=None,
        device=device,
        amp_enabled=args.amp,
        scaler=None,
        log_every=0,
    )
    _, y_val, p_val, prob_val = run_epoch(
        dataloader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=None,
        device=device,
        amp_enabled=args.amp,
        scaler=None,
        log_every=0,
    )
    _, y_test, p_test, prob_test = run_epoch(
        dataloader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=None,
        device=device,
        amp_enabled=args.amp,
        scaler=None,
        log_every=0,
    )

    _, _ = save_split_outputs(
        split_name="train",
        y_true=y_train,
        y_pred=p_train,
        labels=labels,
        label_names=label_names,
        out_dir=out_dir,
        y_prob=prob_train,
        df_meta=train_df,
    )
    _, _ = save_split_outputs(
        split_name="val",
        y_true=y_val,
        y_pred=p_val,
        labels=labels,
        label_names=label_names,
        out_dir=out_dir,
        y_prob=prob_val,
        df_meta=val_df,
    )
    test_summary, _ = save_split_outputs(
        split_name="test",
        y_true=y_test,
        y_pred=p_test,
        labels=labels,
        label_names=label_names,
        out_dir=out_dir,
        y_prob=prob_test,
        df_meta=test_df,
    )

    val_cm = confusion_matrix(y_val, p_val, labels=labels)
    test_cm = confusion_matrix(y_test, p_test, labels=labels)
    np.save(out_dir / "confusion_val.npy", val_cm)
    np.save(out_dir / "confusion_test.npy", test_cm)

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    for split_name, cm in [("val", val_cm), ("test", test_cm)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(include_values=False, cmap="Blues", ax=ax, xticks_rotation=90)
        plt.title(f"{split_name.upper()} Confusion Matrix ({args.backbone})")
        plt.tight_layout()
        fig.savefig(out_dir / f"confusion_{split_name}.png", dpi=220)
        plt.close(fig)

    run_timestamp_utc = _utc_now()
    run_id = args.run_id or f"{out_dir.name}_{run_timestamp_utc.replace(':', '').replace('-', '')}"
    split_hash_path = limuc_root / "0_dataset_prep" / "out" / "metadata" / "split_hash.txt"
    run_meta = {
        "model": "supervised_backbone_finetune",
        "backbone": args.backbone,
        "loss": args.loss,
        "focal_gamma": float(args.focal_gamma),
        "label_smoothing": float(args.label_smoothing),
        "class_weighting": args.class_weighting,
        "scheduler": args.scheduler,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "amp": bool(args.amp),
        "image_size": int(args.image_size),
        "resize_size": int(args.resize_size),
        "aug_strength": args.aug_strength,
        "best_val_macro_f1": float(best_val_macro_f1),
        "split_hash": _read_split_hash(split_hash_path),
        "run_id": run_id,
        "timestamp_utc": run_timestamp_utc,
        "out_dir": str(out_dir.resolve()),
        "script_path": str(Path(__file__).resolve()),
        "limuc_root": str(limuc_root.resolve()),
        "metrics_test": test_summary,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print("[train_supervised] complete")
    print(f"[train_supervised] run_id={run_id}")
    print(f"[train_supervised] out_dir={out_dir}")
    print(f"[train_supervised] test_summary={json.dumps(test_summary, indent=2)}")


if __name__ == "__main__":
    main()
