#!/usr/bin/env python3
"""Build Mode-1 controlled outputs from existing persisted prediction CSVs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def _parse_score(text: str) -> tuple[int, bool]:
    if not isinstance(text, str):
        return -1, False
    m = re.search(r"SCORE:\s*([0-3])", text, flags=re.IGNORECASE)
    if not m:
        return -1, False
    return int(m.group(1)), True


def _balanced_acc(cm: np.ndarray) -> float:
    recalls = []
    for i in range(cm.shape[0]):
        support = cm[i, :].sum()
        recalls.append(float(cm[i, i] / support) if support > 0 else 0.0)
    return float(np.mean(recalls))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, parse_ok: np.ndarray) -> dict:
    y_pred_eval = y_pred.copy()
    y_pred_eval[~parse_ok] = 0
    cm = confusion_matrix(y_true, y_pred_eval, labels=[0, 1, 2, 3])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred_eval)),
        "macro_f1": float(f1_score(y_true, y_pred_eval, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "balanced_accuracy": float(_balanced_acc(cm)),
        "qwk": float(__import__("sklearn.metrics").metrics.cohen_kappa_score(y_true, y_pred_eval, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_pred_eval)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_eval))),
        "parse_rate": float(parse_ok.mean() if len(parse_ok) else 0.0),
    }


def _save_plots(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion (mode1 from persisted)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_test.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [0, 1, 2, 3]
    ys = [int((y_pred == x).sum()) for x in xs]
    ax.bar(xs, ys)
    ax.set_title("Predicted Label Distribution (mode1 from persisted)")
    ax.set_xlabel("Predicted Mayo score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_dir / "pred_label_histogram.png", dpi=220)
    plt.close(fig)


def _parser_audit(pred_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["image_id", "true_label", "raw_generation", "pred_label", "parse_ok"]
    work = pred_df[cols].copy()
    if len(work) <= 20:
        return work
    parts = []
    per_class = 5
    for c in [0, 1, 2, 3]:
        sub = work[work["true_label"] == c]
        if len(sub) > 0:
            parts.append(sub.sample(n=min(per_class, len(sub)), random_state=42))
    used = pd.concat(parts).drop_duplicates() if parts else work.head(0)
    rem = work.drop(index=used.index, errors="ignore")
    need = max(20 - len(used), 0)
    if need > 0 and len(rem) > 0:
        used = pd.concat([used, rem.sample(n=min(need, len(rem)), random_state=42)])
    return used.head(20).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(args.input_csv.resolve())
    run_id = f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    if "raw_text" in src.columns:
        raw = src["raw_text"].astype(str).tolist()
        parsed = [_parse_score(t) for t in raw]
        pred = [p[0] for p in parsed]
        parse_ok = [p[1] for p in parsed]
    else:
        # Persisted run does not include raw generations; fallback from parsed label.
        pred = src["y_pred"].astype(int).tolist()
        parse_ok = [p in [0, 1, 2, 3] for p in pred]
        raw = [f"SCORE: {p}" if ok else "" for p, ok in zip(pred, parse_ok)]

    pred_df = pd.DataFrame(
        {
            "image_id": src["img_id"] if "img_id" in src.columns else np.arange(len(src)),
            "true_label": src["y_true"].astype(int),
            "pred_label": np.array(pred).astype(int),
            "parse_ok": np.array(parse_ok).astype(bool),
            "raw_generation": raw,
            "split": args.split,
            "run_id": run_id,
        }
    )

    pred_df.to_csv(out_dir / f"pred_{args.split}.csv", index=False)
    y_true = pred_df["true_label"].to_numpy(dtype=int)
    y_pred = pred_df["pred_label"].to_numpy(dtype=int)
    parse_ok_arr = pred_df["parse_ok"].to_numpy(dtype=bool)
    y_pred_eval = y_pred.copy()
    y_pred_eval[~parse_ok_arr] = 0

    metrics = _metrics(y_true=y_true, y_pred=y_pred, parse_ok=parse_ok_arr)
    (out_dir / "metrics_test.json").write_text(json.dumps({"split": args.split, "summary": metrics}, indent=2), encoding="utf-8")
    _save_plots(y_true=y_true, y_pred=y_pred_eval, out_dir=out_dir)
    _parser_audit(pred_df).to_csv(out_dir / "parser_audit_samples.csv", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_input_csv": str(args.input_csv.resolve()),
                "mode": "mode1_from_persisted_predictions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out_dir}")
    print(f"Rows: {len(pred_df)}")


if __name__ == "__main__":
    main()
