#!/usr/bin/env python3
"""Train a BLIP2-style VLM with LoRA for LIMUC Mayo severity (0-3).

This script is notebook-equivalent but reproducible from CLI and adds
class-balanced sampling to mitigate majority-class collapse.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("USE_TF", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor, Trainer, TrainingArguments

try:
    from transformers import AutoModelForVision2Seq

    _HAS_VISION2SEQ = True
except Exception:
    from transformers import Blip2ForConditionalGeneration

    AutoModelForVision2Seq = None
    _HAS_VISION2SEQ = False

from peft import LoraConfig, get_peft_model

PROMPT = (
    "You are a medical expert. Given this colonoscopy image, provide the Mayo Endoscopic Score (0-3). "
    "Respond with only: 'SCORE: X' where X is 0,1,2, or 3."
)
VALID_LABELS = [0, 1, 2, 3]


@dataclass
class RunContext:
    data_root: Path
    out_dir: Path
    adapter_dir: Path
    meta_csv: Path
    split_hash: Optional[str]


def json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return repr(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Prototyping_reformat/DatasetAnalysis/LIMUC"),
        help="Path to LIMUC dataset-analysis root (contains 0_dataset_prep, 3_vlm_severity).",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Override metadata_enriched.csv path. Defaults to <data-root>/0_dataset_prep/out/metadata/metadata_enriched.csv",
    )
    parser.add_argument("--model-name", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-fast-processor", action="store_true")
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        default=False,
        help="Enable class-balanced train sampling via WeightedRandomSampler.",
    )

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run inference/metrics.")
    return parser.parse_args()


def resolve_context(args: argparse.Namespace) -> RunContext:
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")

    meta_csv = args.meta_csv.resolve() if args.meta_csv else (data_root / "0_dataset_prep" / "out" / "metadata" / "metadata_enriched.csv")
    if not meta_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {meta_csv}")

    run_name = args.run_name or f"vlm_lora_finetune_mayo_balanced_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = (data_root / "3_vlm_severity" / "results" / run_name).resolve()
    adapter_dir = out_dir / "lora_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_hash_path = data_root / "0_dataset_prep" / "out" / "metadata" / "split_hash.txt"
    split_hash = split_hash_path.read_text(encoding="utf-8").strip() if split_hash_path.exists() else None

    args.run_name = run_name
    return RunContext(
        data_root=data_root,
        out_dir=out_dir,
        adapter_dir=adapter_dir,
        meta_csv=meta_csv,
        split_hash=split_hash,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return (data_root / "0_dataset_prep" / p).resolve()


def load_splits(ctx: RunContext, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    df = pd.read_csv(ctx.meta_csv)
    if "label_id" not in df.columns:
        if "label_name" not in df.columns:
            raise RuntimeError("Metadata must include label_id or label_name")
        names = sorted(df["label_name"].dropna().unique().tolist())
        name_to_id = {n: i for i, n in enumerate(names)}
        df["label_id"] = df["label_name"].map(name_to_id).astype(int)

    if "label_name" in df.columns:
        id_to_name = {}
        for rid, name in zip(df["label_id"], df["label_name"]):
            if int(rid) not in id_to_name:
                id_to_name[int(rid)] = str(name)
    else:
        id_to_name = {i: f"Mayo {i}" for i in VALID_LABELS}

    df["image_path"] = df["image_path"].map(lambda p: str(resolve_image_path(ctx.data_root, str(p))))
    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"].isin(["val", "validation"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if args.max_train_samples > 0:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed).reset_index(drop=True)
    if args.max_val_samples > 0:
        val_df = val_df.sample(n=min(args.max_val_samples, len(val_df)), random_state=args.seed).reset_index(drop=True)
    if args.max_test_samples > 0:
        test_df = test_df.sample(n=min(args.max_test_samples, len(test_df)), random_state=args.seed).reset_index(drop=True)

    return train_df, val_df, test_df, id_to_name


def infer_lora_targets(module: torch.nn.Module) -> List[str]:
    candidates = {"q_proj", "v_proj", "k_proj", "o_proj", "q", "v", "k", "o", "fc1", "fc2", "dense"}
    found: set[str] = set()
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidates:
                found.add(suffix)
    return sorted(found) if found else ["q_proj", "v_proj"]


def load_model_and_processor(args: argparse.Namespace):
    if args.force_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else None)

    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=args.use_fast_processor)
    if model_dtype is not None:
        if _HAS_VISION2SEQ:
            model = AutoModelForVision2Seq.from_pretrained(args.model_name, torch_dtype=model_dtype)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=model_dtype)
    else:
        if _HAS_VISION2SEQ:
            model = AutoModelForVision2Seq.from_pretrained(args.model_name)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)

    if not hasattr(model, "language_model"):
        raise RuntimeError(f"Expected BLIP2-like model with language_model; got {model.__class__.__name__}")

    target_modules = infer_lora_targets(model.language_model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return processor, model, device, use_bf16, target_modules


class VQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, prompt: str):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert("RGB")

        inputs = self.processor(images=img, text=self.prompt, return_tensors="pt")
        if "inputs_embeds" in inputs:
            inputs.pop("inputs_embeds")

        target = f"SCORE: {int(row.label_id)}"
        labels = self.processor.tokenizer(target, return_tensors="pt").input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        item = {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        if "attention_mask" in inputs:
            item["attention_mask"] = inputs["attention_mask"].squeeze(0)
        return item


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]], pad_token_id: int):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)

    out = {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}
    if "attention_mask" in batch[0]:
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        out["attention_mask"] = attention_mask
    return out


def make_weighted_sampler(train_df: pd.DataFrame) -> Tuple[WeightedRandomSampler, Dict[int, int], Dict[int, float]]:
    labels = train_df["label_id"].astype(int).tolist()
    class_counts = Counter(labels)
    class_weights = {c: 1.0 / max(class_counts.get(c, 1), 1) for c in sorted(class_counts.keys())}
    sample_weights = torch.tensor([class_weights[int(y)] for y in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, {int(k): int(v) for k, v in class_counts.items()}, {int(k): float(v) for k, v in class_weights.items()}


class SafeTrainer(Trainer):
    def __init__(self, *args, weighted_sampler: Optional[WeightedRandomSampler] = None, num_workers: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._weighted_sampler = weighted_sampler
        self._num_workers = num_workers

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("inputs_embeds", None)
        inputs.pop("decoder_inputs_embeds", None)
        inputs.pop("num_items_in_batch", None)
        kwargs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        if self._weighted_sampler is None:
            return super().get_train_dataloader()

        data_collator = self.data_collator
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._weighted_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self._num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def parse_score(text: str) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"\b(?:SCORE\s*:\s*)?([0-3])\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def run_split(df: pd.DataFrame, model, processor, device: torch.device, max_new_tokens: int) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    y_true: List[int] = []
    y_pred: List[int] = []
    raw_texts: List[str] = []
    parse_ok: List[bool] = []

    model.eval()
    for row in df.itertuples(index=False):
        img = Image.open(row.image_path).convert("RGB")
        enc = processor(images=img, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(gen, skip_special_tokens=True)[0]
        score = parse_score(text)
        raw_texts.append(text)
        y_true.append(int(row.label_id))
        if score is None:
            y_pred.append(-1)
            parse_ok.append(False)
        else:
            y_pred.append(int(score))
            parse_ok.append(True)

    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), raw_texts, np.asarray(parse_ok, dtype=bool)


def pred_hist(y_pred_eval: np.ndarray, out_path: Path) -> None:
    xs = [0, 1, 2, 3]
    ys = [int((y_pred_eval == x).sum()) for x in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xlabel("Predicted Mayo score")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Label Histogram")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_confusion(y_true_eval: np.ndarray, y_pred_eval: np.ndarray, out_png: Path, out_npy: Path, title: str) -> None:
    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=VALID_LABELS)
    np.save(out_npy, cm)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(VALID_LABELS)
    ax.set_yticklabels(VALID_LABELS)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _sample_parser_audit(pred_df: pd.DataFrame, n_total: int = 20) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    parts: List[pd.DataFrame] = []
    per_class = max(n_total // 4, 1)
    for c in VALID_LABELS:
        sub = pred_df[pred_df["y_true"] == c]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=42))
    used = pd.concat(parts).drop_duplicates() if parts else pd.DataFrame(columns=pred_df.columns)
    remain = pred_df.drop(index=used.index, errors="ignore")
    need = max(n_total - len(used), 0)
    if need > 0 and len(remain) > 0:
        used = pd.concat([used, remain.sample(n=min(need, len(remain)), random_state=42)])
    return used.head(n_total).reset_index(drop=True)


def compute_summary(y_true: np.ndarray, y_pred: np.ndarray, parse_ok: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], np.ndarray]:
    y_eval = y_pred.copy()
    invalid = ~parse_ok
    y_eval[invalid] = 0

    summary = {
        "accuracy": float(accuracy_score(y_true, y_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_eval)),
        "macro_f1": float(f1_score(y_true, y_eval, average="macro", labels=VALID_LABELS, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_eval, average="weighted", labels=VALID_LABELS, zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_eval, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_eval)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_eval))),
        "parse_rate": float(parse_ok.mean() if len(parse_ok) else 0.0),
        "n_invalid": int(invalid.sum()),
    }

    rep = classification_report(
        y_true,
        y_eval,
        labels=VALID_LABELS,
        output_dict=True,
        zero_division=0,
    )

    return summary, rep, y_eval


def save_split_outputs(
    split_name: str,
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_text: Sequence[str],
    parse_ok: np.ndarray,
    out_dir: Path,
) -> Dict[str, object]:
    summary, rep, y_eval = compute_summary(y_true, y_pred, parse_ok)

    metrics_payload = {"split": split_name, "summary": summary, "report": rep}
    (out_dir / f"metrics_{split_name}.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "img_id": df_meta["img_id"].values,
            "image_path": df_meta["image_path"].values,
            "raw_text": list(raw_text),
            "parse_ok": parse_ok.astype(bool),
        }
    )
    pred_df.to_csv(out_dir / f"pred_{split_name}.csv", index=False)
    pred_df.to_csv(out_dir / f"pred_{split_name}_raw.csv", index=False)

    per_class = {k: v for k, v in rep.items() if str(k) in {"0", "1", "2", "3"}}
    pd.DataFrame(per_class).T.to_csv(out_dir / f"per_class_{split_name}.csv")

    save_confusion(
        y_true_eval=y_true,
        y_pred_eval=y_eval,
        out_png=out_dir / f"confusion_{split_name}.png",
        out_npy=out_dir / f"confusion_{split_name}.npy",
        title=f"{split_name.upper()} Confusion Matrix (VLM LoRA finetune)",
    )

    if split_name == "test":
        pred_hist(y_eval, out_dir / "pred_label_histogram.png")
        audit = _sample_parser_audit(pred_df[["img_id", "y_true", "raw_text", "y_pred", "parse_ok"]].copy(), n_total=20)
        audit.rename(columns={"img_id": "image_id", "y_true": "true_label", "y_pred": "pred_label", "raw_text": "raw_generation"}, inplace=True)
        audit.to_csv(out_dir / "parser_audit_samples.csv", index=False)

    return metrics_payload


def main() -> None:
    args = parse_args()
    ctx = resolve_context(args)
    seed_all(args.seed)

    train_df, val_df, test_df, id_to_name = load_splits(ctx, args)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("One or more splits are empty after filtering.")

    processor, model, device, use_bf16, target_modules = load_model_and_processor(args)

    train_ds = VQADataset(train_df, processor, PROMPT)
    val_ds = VQADataset(val_df, processor, PROMPT)

    sampler = None
    class_counts: Dict[int, int] = {}
    class_weights: Dict[int, float] = {}
    if args.balanced_sampling:
        sampler, class_counts, class_weights = make_weighted_sampler(train_df)

    training_args = TrainingArguments(
        output_dir=str(ctx.out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=False,
        bf16=use_bf16,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate_fn(b, pad_token_id=processor.tokenizer.pad_token_id),
        weighted_sampler=sampler,
        num_workers=args.num_workers,
    )

    train_result = None
    if not args.eval_only:
        train_result = trainer.train()

    hist_df = pd.DataFrame(trainer.state.log_history)
    hist_df.to_csv(ctx.out_dir / "training_history.csv", index=False)
    train_summary = {
        "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
        "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_runtime": None if train_result is None else train_result.metrics.get("train_runtime"),
        "train_samples_per_second": None if train_result is None else train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": None if train_result is None else train_result.metrics.get("train_steps_per_second"),
        "train_loss": None if train_result is None else train_result.metrics.get("train_loss"),
    }
    (ctx.out_dir / "training_summary.json").write_text(
        json.dumps(json_safe(train_summary), indent=2),
        encoding="utf-8",
    )

    ctx.adapter_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "language_model") and hasattr(model.language_model, "save_pretrained"):
        model.language_model.save_pretrained(ctx.adapter_dir, safe_serialization=True)
    else:
        model.save_pretrained(ctx.adapter_dir, safe_serialization=True)
    processor.save_pretrained(ctx.adapter_dir)

    peft_config = {}
    if hasattr(model, "language_model") and hasattr(model.language_model, "peft_config"):
        for key, cfg in model.language_model.peft_config.items():
            if hasattr(cfg, "to_dict"):
                peft_config[str(key)] = cfg.to_dict()
            else:
                peft_config[str(key)] = {"repr": repr(cfg)}
    (ctx.out_dir / "lora_config.json").write_text(
        json.dumps(json_safe(peft_config), indent=2),
        encoding="utf-8",
    )

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    lora_param_counts = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": float(100.0 * trainable_params / max(total_params, 1)),
    }
    (ctx.out_dir / "lora_param_count.json").write_text(
        json.dumps(json_safe(lora_param_counts), indent=2),
        encoding="utf-8",
    )

    y_val, pred_val, text_val, parse_val = run_split(
        val_df, model=model, processor=processor, device=device, max_new_tokens=args.max_new_tokens
    )
    y_test, pred_test, text_test, parse_test = run_split(
        test_df, model=model, processor=processor, device=device, max_new_tokens=args.max_new_tokens
    )

    val_metrics = save_split_outputs(
        split_name="val",
        df_meta=val_df,
        y_true=y_val,
        y_pred=pred_val,
        raw_text=text_val,
        parse_ok=parse_val,
        out_dir=ctx.out_dir,
    )
    test_metrics = save_split_outputs(
        split_name="test",
        df_meta=test_df,
        y_true=y_test,
        y_pred=pred_test,
        raw_text=text_test,
        parse_ok=parse_test,
        out_dir=ctx.out_dir,
    )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_meta = {
        "model": "vlm_lora_finetune",
        "model_name": args.model_name,
        "prompt": PROMPT,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": target_modules,
        "balanced_sampling": bool(args.balanced_sampling),
        "class_counts_train": class_counts,
        "class_weights_sampler": class_weights,
        "split_hash": ctx.split_hash,
        "run_id": run_id,
        "timestamp_utc": run_timestamp,
        "out_dir": str(ctx.out_dir),
        "adapter_dir": str(ctx.adapter_dir),
        "lora_adapter_path": str(ctx.adapter_dir),
        "script_path": str(Path(__file__).resolve()),
        "meta_csv": str(ctx.meta_csv),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "max_train_samples": int(args.max_train_samples),
        "max_val_samples": int(args.max_val_samples),
        "max_test_samples": int(args.max_test_samples),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "metrics_val": val_metrics.get("summary", {}),
        "metrics_test": test_metrics.get("summary", {}),
    }
    (ctx.out_dir / "run_meta.json").write_text(json.dumps(json_safe(run_meta), indent=2), encoding="utf-8")

    print("Saved outputs to", ctx.out_dir)
    print("Val summary:", json.dumps(val_metrics.get("summary", {}), indent=2))
    print("Test summary:", json.dumps(test_metrics.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
