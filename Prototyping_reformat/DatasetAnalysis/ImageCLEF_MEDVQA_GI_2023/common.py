from __future__ import annotations

from pathlib import Path
import json
import hashlib
from typing import Dict, Iterable

import numpy as np
import pandas as pd

OOV_TOKEN = "__OTHER__"

NA_ALIASES = {
    "", "n/a", "na", "not applicable", "none", "no relevant answer",
    "not relevant", "unknown", "unk", "null", "nil", "-",
}


def normalize_answer(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip().lower()
    s = " ".join(s.replace("\n", " ").replace("\t", " ").split())
    if s in NA_ALIASES:
        return "na"
    return s


def find_long_table(root: Path) -> Path:
    candidates = [
        root / "0_dataset_prep" / "out" / "long_table.parquet",
        root / "0_dataset_prep" / "out" / "long_table.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find long_table.parquet or long_table.csv in 0_dataset_prep/out."
    )


def load_long_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                df = pd.read_csv(csv_fallback)
            else:
                raise
    elif path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported long table format: {path}")
    required = {"image_id", "image_path", "question_id", "question_text", "answer_raw", "answer_norm", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Long table missing columns: {sorted(missing)}")
    return df


def build_label_maps(
    df: pd.DataFrame,
    qid_col: str = "question_id",
    ans_col: str = "answer_norm",
    split_col: str = "split",
    oov_token: str = OOV_TOKEN,
) -> Dict[str, Dict[str, Dict]]:
    if split_col in df.columns:
        train_df = df[df[split_col] == "train"]
        if len(train_df) == 0:
            train_df = df
    else:
        train_df = df

    label_maps: Dict[str, Dict[str, Dict]] = {}
    for qid, g in train_df.groupby(qid_col):
        answers = sorted({normalize_answer(a) for a in g[ans_col].tolist()})
        if oov_token not in answers:
            answers.append(oov_token)
        answer_to_id = {a: i for i, a in enumerate(answers)}
        id_to_answer = {i: a for a, i in answer_to_id.items()}
        label_maps[str(qid)] = {"answer_to_id": answer_to_id, "id_to_answer": id_to_answer}
    return label_maps


def save_label_maps(label_maps: Dict[str, Dict[str, Dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (qid, maps) in enumerate(label_maps.items()):
        df = pd.DataFrame(
            {
                "question_id": [qid] * len(maps["answer_to_id"]),
                "answer": list(maps["answer_to_id"].keys()),
                "class_id": list(maps["answer_to_id"].values()),
            }
        )
        qhash = hashlib.md5(str(qid).encode("utf-8")).hexdigest()[:10]
        safe_name = f"{idx:02d}_{qhash}"
        df.to_csv(out_dir / f"label_map_qid_{safe_name}.csv", index=False)


def load_label_maps(label_dir: Path) -> Dict[str, Dict[str, Dict]]:
    if not label_dir.exists():
        raise FileNotFoundError(f"Label map directory not found: {label_dir}")
    label_maps: Dict[str, Dict[str, Dict]] = {}
    for path in sorted(label_dir.glob("label_map_qid_*.csv")):
        df = pd.read_csv(path)
        if "question_id" in df.columns:
            qid = str(df["question_id"].iloc[0])
        else:
            qid = path.stem.replace("label_map_qid_", "")
        answer_to_id = {str(a): int(i) for a, i in zip(df["answer"], df["class_id"])}
        id_to_answer = {int(i): str(a) for a, i in answer_to_id.items()}
        label_maps[qid] = {"answer_to_id": answer_to_id, "id_to_answer": id_to_answer}
    if not label_maps:
        raise FileNotFoundError(f"No label_map_qid_*.csv files in {label_dir}")
    return label_maps


def add_label_ids(
    df: pd.DataFrame,
    label_maps: Dict[str, Dict[str, Dict]],
    qid_col: str = "question_id",
    ans_col: str = "answer_norm",
    out_col: str = "label_id",
    oov_token: str = OOV_TOKEN,
) -> pd.DataFrame:
    df = df.copy()
    labels = np.full(len(df), -1, dtype=int)
    for qid, g in df.groupby(qid_col):
        maps = label_maps.get(str(qid))
        if maps is None:
            raise KeyError(f"Missing label map for question_id={qid}")
        ans_to_id = maps["answer_to_id"]
        oov_id = ans_to_id.get(oov_token, -1)
        labels[g.index] = g[ans_col].map(ans_to_id).fillna(oov_id).astype(int).values
    df[out_col] = labels
    return df


def add_answer_from_ids(
    df: pd.DataFrame,
    label_maps: Dict[str, Dict[str, Dict]],
    id_col: str = "pred_label_id",
    out_col: str = "pred_answer",
    qid_col: str = "question_id",
) -> pd.DataFrame:
    df = df.copy()
    answers = [None] * len(df)
    for qid, g in df.groupby(qid_col):
        maps = label_maps.get(str(qid))
        if maps is None:
            raise KeyError(f"Missing label map for question_id={qid}")
        id_to_answer = maps["id_to_answer"]
        for idx, v in g[id_col].items():
            answers[idx] = id_to_answer.get(int(v), OOV_TOKEN)
    df[out_col] = answers
    return df


try:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    _HAS_SK = True
except Exception:
    accuracy_score = f1_score = precision_recall_fscore_support = None
    _HAS_SK = False


def _macro_f1_numpy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        if tp == 0 and (fp == 0 or fn == 0):
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_basic_metrics(y_true, y_pred) -> Dict[str, float]:
    if _HAS_SK:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {"accuracy": acc, "macro_f1": _macro_f1_numpy(y_true, y_pred)}


def compute_metrics_per_question(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    qid_col: str = "question_id",
) -> tuple[Dict[str, float], pd.DataFrame]:
    per_q = []
    for qid, g in df.groupby(qid_col):
        metrics = compute_basic_metrics(g[y_true_col], g[y_pred_col])
        per_q.append({"question_id": qid, "n": int(len(g)), **metrics})
    overall = compute_basic_metrics(df[y_true_col], df[y_pred_col])
    overall["n"] = int(len(df))
    per_q_df = pd.DataFrame(per_q).sort_values("question_id")
    return overall, per_q_df


def binary_question_ids(label_maps: Dict[str, Dict[str, Dict]]) -> list[str]:
    qids = []
    for qid, maps in label_maps.items():
        labels = [a for a in maps["answer_to_id"].keys() if a != OOV_TOKEN]
        if len(labels) == 2:
            qids.append(qid)
    return qids


def compute_binary_metrics(
    df: pd.DataFrame,
    label_maps: Dict[str, Dict[str, Dict]],
    y_true_col: str,
    y_pred_col: str,
    qid_col: str = "question_id",
) -> pd.DataFrame:
    rows = []
    for qid in binary_question_ids(label_maps):
        g = df[df[qid_col] == qid]
        if len(g) == 0:
            continue
        if _HAS_SK:
            prec, rec, f1, _ = precision_recall_fscore_support(
                g[y_true_col], g[y_pred_col], average="macro", zero_division=0
            )
        else:
            # Macro precision/recall for binary case
            labels = sorted(set(g[y_true_col].tolist()) | set(g[y_pred_col].tolist()))
            precs, recs = [], []
            for label in labels:
                tp = int(((g[y_true_col] == label) & (g[y_pred_col] == label)).sum())
                fp = int(((g[y_true_col] != label) & (g[y_pred_col] == label)).sum())
                fn = int(((g[y_true_col] == label) & (g[y_pred_col] != label)).sum())
                precs.append(tp / (tp + fp) if (tp + fp) else 0.0)
                recs.append(tp / (tp + fn) if (tp + fn) else 0.0)
            prec = float(np.mean(precs)) if precs else 0.0
            rec = float(np.mean(recs)) if recs else 0.0
            f1 = float(_macro_f1_numpy(g[y_true_col], g[y_pred_col]))
        rows.append({
            "question_id": qid,
            "n": int(len(g)),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
        })
    return pd.DataFrame(rows).sort_values("question_id")


def save_metrics(
    out_dir: Path,
    overall: Dict[str, float],
    per_question: pd.DataFrame,
    binary: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_overall.json", "w") as f:
        json.dump(overall, f, indent=2)
    per_question.to_csv(out_dir / "metrics_per_question.csv", index=False)
    if binary is not None and len(binary) > 0:
        binary.to_csv(out_dir / "metrics_binary.csv", index=False)


def save_predictions(df: pd.DataFrame, out_dir: Path, filename: str = "predictions.csv", columns=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df[columns]
    df.to_csv(out_dir / filename, index=False)
