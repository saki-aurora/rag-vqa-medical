import re
import string
from collections import Counter
from typing import Iterable, Dict, List, Tuple

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    t = "" if text is None else str(text)
    t = t.lower()
    t = t.translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def anls(pred: str, gold: str, threshold: float = 0.5) -> float:
    """Approximate ANLS using normalized Levenshtein similarity."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0

    dist = _levenshtein(p, g)
    score = 1.0 - (dist / max_len)
    return score if score >= threshold else 0.0


def compute_metrics(
    preds: Iterable[str],
    golds: Iterable[str],
    anls_threshold: float = 0.5,
) -> Dict[str, float]:
    preds = list(preds)
    golds = list(golds)
    if len(preds) != len(golds):
        raise ValueError(f"Pred/gold length mismatch: {len(preds)} vs {len(golds)}")

    em_scores = []
    f1_scores = []
    anls_scores = []

    for p, g in zip(preds, golds):
        em_scores.append(exact_match(p, g))
        f1_scores.append(token_f1(p, g))
        anls_scores.append(anls(p, g, threshold=anls_threshold))

    n = len(preds) if preds else 1
    return {
        "em": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
        "anls": sum(anls_scores) / n,
        "count": len(preds),
    }


def group_metrics(
    df,
    pred_col: str,
    gold_col: str,
    group_col: str,
    anls_threshold: float = 0.5,
):
    rows = []
    for key, g in df.groupby(group_col):
        m = compute_metrics(g[pred_col].tolist(), g[gold_col].tolist(), anls_threshold)
        m[group_col] = key
        rows.append(m)
    return rows
