#!/usr/bin/env python3
"""Controlled generative UC Mayo evaluation with two inference modes.

Mode 1: free generation + strict SCORE parser
Mode 2: label scoring via next-token probabilities after "SCORE:"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
    _HAS_VISION2SEQ = True
except Exception:
    from transformers import Blip2ForConditionalGeneration
    AutoModelForVision2Seq = None
    _HAS_VISION2SEQ = False

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False


PROMPT = (
    "You are a medical imaging assistant. Rate ulcerative colitis severity using the Mayo endoscopic subscore. "
    "Output EXACTLY in this format: SCORE: <0|1|2|3>"
)

CANONICAL_SPLIT_HASH = "d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model-name", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--mode", type=str, choices=["mode1", "mode2", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--processor-use-fast", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _resolve_image_path(meta_csv: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    # metadata expected at .../0_dataset_prep/out/metadata/metadata_enriched.csv
    base_0_dataset_prep = meta_csv.parent.parent.parent
    return (base_0_dataset_prep / pp).resolve()


def load_split_df(meta_csv: Path, split: str, max_samples: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    if "split" not in df.columns:
        raise RuntimeError("metadata must include 'split' column")
    if "img_id" not in df.columns:
        raise RuntimeError("metadata must include 'img_id' column")
    if "label_id" not in df.columns:
        if "label_name" not in df.columns:
            raise RuntimeError("metadata must include label_id or label_name")
        sorted_labels = sorted(df["label_name"].dropna().unique().tolist())
        name_to_id = {name: i for i, name in enumerate(sorted_labels)}
        df["label_id"] = df["label_name"].map(name_to_id).astype(int)

    df = df[df["split"] == split].reset_index(drop=True).copy()
    df["resolved_image_path"] = df["image_path"].map(lambda p: str(_resolve_image_path(meta_csv, str(p))))
    df = df[df["resolved_image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    if max_samples and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)
    return df


def load_model(model_name: str, force_cuda: bool, adapter_dir: Path | None, processor_use_fast: bool):
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else None)

    processor = AutoProcessor.from_pretrained(model_name, use_fast=processor_use_fast)
    if model_dtype is not None:
        if _HAS_VISION2SEQ:
            model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=model_dtype)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=model_dtype)
    else:
        if _HAS_VISION2SEQ:
            model = AutoModelForVision2Seq.from_pretrained(model_name)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    adapter_loaded = False
    if adapter_dir is not None:
        if not _HAS_PEFT:
            raise RuntimeError("peft is required to load adapter-dir but is not installed.")
        if not adapter_dir.exists():
            raise RuntimeError(f"Adapter directory not found: {adapter_dir}")
        weight_files = list(adapter_dir.glob("adapter_model.*"))
        if not weight_files:
            raise RuntimeError(f"No adapter weights found under: {adapter_dir}")
        if not hasattr(model, "language_model"):
            raise RuntimeError("Expected BLIP2 model with language_model for LoRA adapter load.")
        model.language_model = PeftModel.from_pretrained(model.language_model, str(adapter_dir))
        adapter_loaded = True

    model.eval()
    return processor, model, device, adapter_loaded


def parse_mode1_score(text: str) -> Tuple[int, bool]:
    if text is None:
        return -1, False
    m = re.search(r"SCORE:\s*([0-3])", text, flags=re.IGNORECASE)
    if not m:
        return -1, False
    return int(m.group(1)), True


def _balanced_acc_from_confusion(cm: np.ndarray) -> float:
    recalls: List[float] = []
    for i in range(cm.shape[0]):
        support = cm[i, :].sum()
        recalls.append(float(cm[i, i] / support) if support > 0 else 0.0)
    return float(np.mean(recalls))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, parse_ok: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    parse_ok = np.asarray(parse_ok).astype(bool)

    # Replace invalid predictions with class 0 for ordinal metrics, but expose parse_rate explicitly.
    y_pred_for_ordinal = y_pred.copy()
    y_pred_for_ordinal[~parse_ok] = 0

    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred_for_ordinal, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_for_ordinal)),
        "macro_f1": float(f1_score(y_true, y_pred_for_ordinal, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(_balanced_acc_from_confusion(cm)),
        "qwk": float(
            np.nan_to_num(
                __import__("sklearn.metrics").metrics.cohen_kappa_score(
                    y_true, y_pred_for_ordinal, weights="quadratic"
                )
            )
        ),
        "mae": float(mean_absolute_error(y_true, y_pred_for_ordinal)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred_for_ordinal))),
        "parse_rate": float(parse_ok.mean() if len(parse_ok) else 0.0),
    }
    return metrics


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_hist(y_pred: np.ndarray, out_path: Path, title: str) -> None:
    arr = np.asarray(y_pred).astype(int)
    xs = [0, 1, 2, 3]
    ys = [int((arr == x).sum()) for x in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xlabel("Predicted Mayo score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _sample_parser_audit(df: pd.DataFrame, n_total: int = 20) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    parts: List[pd.DataFrame] = []
    per_class = max(n_total // 4, 1)
    for c in [0, 1, 2, 3]:
        sub = df[df["true_label"] == c]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=42))
    used = pd.concat(parts).drop_duplicates() if parts else pd.DataFrame(columns=df.columns)
    remaining = df.drop(index=used.index, errors="ignore")
    need = max(n_total - len(used), 0)
    if need > 0 and len(remaining) > 0:
        used = pd.concat([used, remaining.sample(n=min(need, len(remaining)), random_state=42)])
    return used.head(n_total).reset_index(drop=True)


def eval_mode1(
    df: pd.DataFrame,
    processor,
    model,
    device,
    run_id: str,
    split: str,
    max_new_tokens: int,
    log_every: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        img = Image.open(row.resolved_image_path).convert("RGB")
        inputs = processor(images=img, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        pred, parse_ok = parse_mode1_score(text)
        rows.append(
            {
                "image_id": row.img_id,
                "true_label": int(row.label_id),
                "pred_label": int(pred),
                "parse_ok": bool(parse_ok),
                "raw_generation": text,
                "split": split,
                "run_id": run_id,
            }
        )
        if log_every > 0 and idx % log_every == 0:
            print(f"[mode1] processed {idx}/{len(df)}")
    return pd.DataFrame(rows)


def eval_mode2(df: pd.DataFrame, processor, model, device, run_id: str, split: str, log_every: int) -> pd.DataFrame:
    tok = processor.tokenizer
    prefix_ids = tok("SCORE:", add_special_tokens=False).input_ids

    candidate_ids: Dict[int, int] = {}
    for c in [0, 1, 2, 3]:
        ids = tok(f" {c}", add_special_tokens=False).input_ids
        if not ids:
            ids = tok(str(c), add_special_tokens=False).input_ids
        if not ids:
            raise RuntimeError(f"Could not tokenize candidate class: {c}")
        candidate_ids[c] = int(ids[0])

    rows: List[Dict[str, object]] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        img = Image.open(row.resolved_image_path).convert("RGB")
        enc = processor(images=img, text=PROMPT, return_tensors="pt").to(device)

        # Build decoder prefix: <decoder_start> SCORE:
        decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
        if decoder_start_id is None:
            decoder_start_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        dec = torch.tensor([[decoder_start_id] + prefix_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(**enc, decoder_input_ids=dec)
            logits = out.logits[:, -1, :]  # token after "SCORE:"
            cand_logits = torch.stack([logits[0, candidate_ids[c]] for c in [0, 1, 2, 3]]).float()
            probs = torch.softmax(cand_logits, dim=0).detach().cpu().numpy().tolist()

        pred = int(np.argmax(probs))
        rows.append(
            {
                "image_id": row.img_id,
                "true_label": int(row.label_id),
                "pred_label": pred,
                "parse_ok": True,
                "raw_generation": f"SCORE: {pred}",
                "p0": float(probs[0]),
                "p1": float(probs[1]),
                "p2": float(probs[2]),
                "p3": float(probs[3]),
                "split": split,
                "run_id": run_id,
            }
        )
        if log_every > 0 and idx % log_every == 0:
            print(f"[mode2] processed {idx}/{len(df)}")
    return pd.DataFrame(rows)


def persist_outputs(pred_df: pd.DataFrame, out_dir: Path, split: str, mode_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"pred_{split}.csv"
    pred_df.to_csv(pred_csv, index=False)

    y_true = pred_df["true_label"].to_numpy(dtype=int)
    y_pred = pred_df["pred_label"].to_numpy(dtype=int)
    parse_ok = pred_df["parse_ok"].to_numpy(dtype=bool)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, parse_ok=parse_ok)
    (out_dir / "metrics_test.json").write_text(
        json.dumps(
            {
                "split": split,
                "mode": mode_name,
                "summary": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    y_pred_for_plot = y_pred.copy()
    y_pred_for_plot[~parse_ok] = 0
    save_confusion(y_true, y_pred_for_plot, out_dir / "confusion_test.png", f"Confusion ({mode_name})")
    save_hist(y_pred_for_plot, out_dir / "pred_label_histogram.png", f"Predictions ({mode_name})")

    if "raw_generation" in pred_df.columns:
        audit_cols = ["image_id", "true_label", "raw_generation", "pred_label", "parse_ok"]
        audit_df = _sample_parser_audit(pred_df[audit_cols].copy(), n_total=20)
        audit_df.to_csv(out_dir / "parser_audit_samples.csv", index=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = args.out_dir.resolve() / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    df = load_split_df(
        meta_csv=args.meta_csv.resolve(),
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if df.empty:
        raise RuntimeError("No rows found for requested split after image-path resolution.")

    processor, model, device, adapter_loaded = load_model(
        model_name=args.model_name,
        force_cuda=args.force_cuda,
        adapter_dir=args.adapter_dir.resolve() if args.adapter_dir else None,
        processor_use_fast=args.processor_use_fast,
    )

    mode1_done = False
    mode2_done = False
    if args.mode in {"mode1", "both"}:
        pred1 = eval_mode1(
            df=df,
            processor=processor,
            model=model,
            device=device,
            run_id=run_id,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            log_every=args.log_every,
        )
        persist_outputs(pred1, out_root / "mode1_free_generation", split=args.split, mode_name="mode1_free_generation")
        mode1_done = True

    if args.mode in {"mode2", "both"}:
        pred2 = eval_mode2(
            df=df,
            processor=processor,
            model=model,
            device=device,
            run_id=run_id,
            split=args.split,
            log_every=args.log_every,
        )
        persist_outputs(pred2, out_root / "mode2_label_scoring", split=args.split, mode_name="mode2_label_scoring")
        mode2_done = True

    run_meta = {
        "run_id": run_id,
        "model_name": args.model_name,
        "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
        "adapter_loaded": adapter_loaded,
        "prompt": PROMPT,
        "split": args.split,
        "n_rows": int(len(df)),
        "split_hash": CANONICAL_SPLIT_HASH,
        "modes": {
            "mode1": mode1_done,
            "mode2": mode2_done,
        },
        "meta_csv": str(args.meta_csv.resolve()),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Wrote run root: {out_root}")
    print(f"Rows: {len(df)}")
    print(f"Mode1 done: {mode1_done}")
    print(f"Mode2 done: {mode2_done}")
    print(f"Adapter loaded: {adapter_loaded}")


if __name__ == "__main__":
    main()
