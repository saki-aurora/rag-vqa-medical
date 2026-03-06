#!/usr/bin/env python3
"""Pass 8: internal LIMUC fusion sweep (fit on val, evaluate on test)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for p in str(raw).split(","):
        s = p.strip()
        if s:
            out.append(s)
    if not out:
        raise ValueError("Expected non-empty comma-separated list.")
    return out


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
            and (nested / "3_vlm_severity").exists()
            and (nested / "4_reporting").exists()
        ):
            return nested
        if (
            (p / "0_dataset_prep").exists()
            and (p / "2_supervised_finetuning").exists()
            and (p / "3_vlm_severity").exists()
            and (p / "4_reporting").exists()
        ):
            return p
    raise RuntimeError(f"Could not locate LIMUC root from start={start_path}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def _argmax_vote(pred_matrix: np.ndarray, n_classes: int = 4) -> np.ndarray:
    out = np.zeros((pred_matrix.shape[0],), dtype=int)
    for i, row in enumerate(pred_matrix):
        out[i] = int(np.bincount(row.astype(int), minlength=n_classes).argmax())
    return out


def _load_pred_csv(path: Path, split: str, *, with_probs: bool) -> pd.DataFrame:
    pred_path = path / f"pred_{split}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)
    req = {"img_id", "y_true", "y_pred"}
    if not req.issubset(set(df.columns)):
        raise RuntimeError(f"Missing required columns in {pred_path}: {sorted(req)}")
    keep = ["img_id", "y_true", "y_pred"]
    if with_probs:
        for c in ["prob_0", "prob_1", "prob_2", "prob_3"]:
            if c not in df.columns:
                raise RuntimeError(f"Expected probability column '{c}' in {pred_path}")
            keep.append(c)
    return df.loc[:, keep].copy()


@dataclass
class SplitTable:
    split: str
    frame: pd.DataFrame
    feature_cols: List[str]


def _build_split_table(
    *,
    limuc_root: Path,
    split: str,
    vlm_runs: Sequence[str],
    res_runs: Sequence[str],
) -> SplitTable:
    base: pd.DataFrame | None = None

    for i, run_name in enumerate(vlm_runs):
        run_dir = limuc_root / "3_vlm_severity" / "results" / run_name
        df = _load_pred_csv(run_dir, split, with_probs=False).rename(columns={"y_pred": f"vlm_pred_{i}"})
        cols = ["img_id", "y_true", f"vlm_pred_{i}"]
        if base is None:
            base = df.loc[:, cols].copy()
        else:
            base = base.merge(df.loc[:, ["img_id", f"vlm_pred_{i}"]], on="img_id", how="inner")

    for i, run_name in enumerate(res_runs):
        run_dir = limuc_root / "2_supervised_finetuning" / "results" / run_name
        df = _load_pred_csv(run_dir, split, with_probs=True).rename(
            columns={
                "y_pred": f"res_pred_{i}",
                "prob_0": f"res_p0_{i}",
                "prob_1": f"res_p1_{i}",
                "prob_2": f"res_p2_{i}",
                "prob_3": f"res_p3_{i}",
            }
        )
        base = base.merge(
            df.loc[:, ["img_id", f"res_pred_{i}", f"res_p0_{i}", f"res_p1_{i}", f"res_p2_{i}", f"res_p3_{i}"]],
            on="img_id",
            how="inner",
        )

    if base is None or base.empty:
        raise RuntimeError(f"No merged rows for split={split}")

    vlm_cols = [f"vlm_pred_{i}" for i in range(len(vlm_runs))]
    res_cols = [f"res_pred_{i}" for i in range(len(res_runs))]

    for i in range(len(res_runs)):
        probs = base.loc[:, [f"res_p0_{i}", f"res_p1_{i}", f"res_p2_{i}", f"res_p3_{i}"]].to_numpy(dtype=float)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        sorted_probs = np.sort(probs, axis=1)
        base[f"res_conf_{i}"] = probs.max(axis=1)
        base[f"res_margin_{i}"] = sorted_probs[:, -1] - sorted_probs[:, -2]

    for c in range(4):
        base[f"vlm_vote_{c}"] = (base.loc[:, vlm_cols] == c).sum(axis=1)
        base[f"res_vote_{c}"] = (base.loc[:, res_cols] == c).sum(axis=1)
        per_seed_prob_cols = [f"res_p{c}_{i}" for i in range(len(res_runs))]
        base[f"res_p{c}_mean"] = base.loc[:, per_seed_prob_cols].mean(axis=1)
        base[f"res_p{c}_std"] = base.loc[:, per_seed_prob_cols].std(axis=1)

    base["vlm_pred_mean"] = base.loc[:, vlm_cols].mean(axis=1)
    base["res_pred_mean"] = base.loc[:, res_cols].mean(axis=1)

    feature_cols = [c for c in base.columns if c not in {"img_id", "y_true"}]
    return SplitTable(split=split, frame=base, feature_cols=feature_cols)


def parse_args() -> argparse.Namespace:
    default_root = find_limuc_root(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limuc-root", type=Path, default=default_root)
    parser.add_argument(
        "--vlm-runs",
        type=str,
        default="vlm_lora_objfix_b200_seed011,vlm_lora_objfix_b200_seed023,vlm_lora_objfix_b200_seed077",
    )
    parser.add_argument(
        "--resnet-runs",
        type=str,
        default="finetune_resnet50_pass5_seed011,finetune_resnet50_pass5_seed023,finetune_resnet50_pass5_seed042",
    )
    parser.add_argument("--tag", type=str, default="pass8_internal_fusion")
    parser.add_argument("--out-dir", type=Path, default=default_root / "4_reporting" / "out")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def _evaluate_candidate(
    *,
    name: str,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    rows: List[Dict[str, Any]],
) -> None:
    mv = _metrics(y_val, pred_val)
    mt = _metrics(y_test, pred_test)
    rows.append(
        {
            "candidate": name,
            "val_accuracy": mv["accuracy"],
            "val_macro_f1": mv["macro_f1"],
            "val_balanced_accuracy": mv["balanced_accuracy"],
            "val_qwk": mv["qwk"],
            "test_accuracy": mt["accuracy"],
            "test_macro_f1": mt["macro_f1"],
            "test_balanced_accuracy": mt["balanced_accuracy"],
            "test_qwk": mt["qwk"],
        }
    )


def main() -> None:
    args = parse_args()
    limuc_root = args.limuc_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_out = out_dir / f"{args.tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_out.mkdir(parents=True, exist_ok=True)

    vlm_runs = _parse_list(args.vlm_runs)
    res_runs = _parse_list(args.resnet_runs)

    val = _build_split_table(limuc_root=limuc_root, split="val", vlm_runs=vlm_runs, res_runs=res_runs)
    test = _build_split_table(limuc_root=limuc_root, split="test", vlm_runs=vlm_runs, res_runs=res_runs)

    common_features = [c for c in val.feature_cols if c in set(test.feature_cols)]
    X_val = val.frame.loc[:, common_features].to_numpy(dtype=float)
    y_val = val.frame["y_true"].astype(int).to_numpy()
    X_test = test.frame.loc[:, common_features].to_numpy(dtype=float)
    y_test = test.frame["y_true"].astype(int).to_numpy()

    rows: List[Dict[str, Any]] = []
    pred_test_map: Dict[str, np.ndarray] = {}

    # Baseline A: mode1 (VLM) majority vote.
    val_vlm = val.frame.loc[:, [f"vlm_pred_{i}" for i in range(len(vlm_runs))]].to_numpy(dtype=int)
    test_vlm = test.frame.loc[:, [f"vlm_pred_{i}" for i in range(len(vlm_runs))]].to_numpy(dtype=int)
    val_pred = _argmax_vote(val_vlm)
    test_pred = _argmax_vote(test_vlm)
    _evaluate_candidate(
        name="baseline_vlm_vote3",
        y_val=y_val,
        pred_val=val_pred,
        y_test=y_test,
        pred_test=test_pred,
        rows=rows,
    )
    pred_test_map["baseline_vlm_vote3"] = test_pred

    # Baseline B: ResNet probability averaging.
    def _res_prob_avg(frame: pd.DataFrame) -> np.ndarray:
        probs: List[np.ndarray] = []
        for i in range(len(res_runs)):
            p = frame.loc[:, [f"res_p0_{i}", f"res_p1_{i}", f"res_p2_{i}", f"res_p3_{i}"]].to_numpy(dtype=float)
            p = np.clip(p, 1e-12, None)
            p = p / p.sum(axis=1, keepdims=True)
            probs.append(p)
        return np.mean(np.stack(probs, axis=0), axis=0).argmax(axis=1).astype(int)

    val_pred = _res_prob_avg(val.frame)
    test_pred = _res_prob_avg(test.frame)
    _evaluate_candidate(
        name="baseline_resnet_probavg3",
        y_val=y_val,
        pred_val=val_pred,
        y_test=y_test,
        pred_test=test_pred,
        rows=rows,
    )
    pred_test_map["baseline_resnet_probavg3"] = test_pred

    # Baseline C: disagreement switch by ResNet confidence.
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80]:
        val_mode1 = _argmax_vote(val.frame.loc[:, [f"vlm_pred_{i}" for i in range(len(vlm_runs))]].to_numpy(dtype=int))
        test_mode1 = _argmax_vote(
            test.frame.loc[:, [f"vlm_pred_{i}" for i in range(len(vlm_runs))]].to_numpy(dtype=int)
        )
        val_res = _res_prob_avg(val.frame)
        test_res = _res_prob_avg(test.frame)
        val_res_conf = (
            val.frame.loc[:, [f"res_conf_{i}" for i in range(len(res_runs))]].to_numpy(dtype=float).mean(axis=1)
        )
        test_res_conf = (
            test.frame.loc[:, [f"res_conf_{i}" for i in range(len(res_runs))]].to_numpy(dtype=float).mean(axis=1)
        )
        val_h = np.where((val_res != val_mode1) & (val_res_conf >= thr), val_res, val_mode1)
        test_h = np.where((test_res != test_mode1) & (test_res_conf >= thr), test_res, test_mode1)
        name = f"hybrid_switch_conf_{thr:.2f}"
        _evaluate_candidate(name=name, y_val=y_val, pred_val=val_h, y_test=y_test, pred_test=test_h, rows=rows)
        pred_test_map[name] = test_h

    # Learned candidates (fit on val only).
    learned_candidates: List[tuple[str, Any]] = []
    for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
        learned_candidates.append(
            (
                f"logreg_C{c}",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=6000, C=c)),
            )
        )
        learned_candidates.append(
            (
                f"logreg_balanced_C{c}",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=6000, C=c, class_weight="balanced")),
            )
        )
    for c, gamma in [(0.3, "scale"), (1.0, "scale"), (3.0, "scale"), (10.0, "scale"), (3.0, 0.03)]:
        learned_candidates.append(
            (
                f"svc_rbf_C{c}_gamma{gamma}",
                make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel="rbf")),
            )
        )

    for name, model in learned_candidates:
        model.fit(X_val, y_val)
        pred_val = model.predict(X_val).astype(int)
        pred_test = model.predict(X_test).astype(int)
        _evaluate_candidate(name=name, y_val=y_val, pred_val=pred_val, y_test=y_test, pred_test=pred_test, rows=rows)
        pred_test_map[name] = pred_test

    df = pd.DataFrame(rows).sort_values(["val_qwk", "val_accuracy"], ascending=[False, False]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    csv_path = run_out / "pass8_internal_fusion_candidates.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "generated_utc": _utc_now(),
        "limuc_root": str(limuc_root),
        "tag": args.tag,
        "vlm_runs": list(vlm_runs),
        "resnet_runs": list(res_runs),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(len(common_features)),
        "best_by_val": best,
        "outputs": {
            "candidates_csv": str(csv_path),
        },
    }

    if args.save_predictions:
        cand = str(best["candidate"])
        pred_df = pd.DataFrame(
            {
                "img_id": test.frame["img_id"].astype(str),
                "y_true": y_test.astype(int),
                "y_pred": pred_test_map[cand].astype(int),
                "candidate": cand,
            }
        )
        pred_path = run_out / "best_candidate_pred_test.csv"
        pred_df.to_csv(pred_path, index=False)
        summary["outputs"]["best_candidate_pred_test_csv"] = str(pred_path)

    json_path = run_out / "pass8_internal_fusion_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Pass 8 Internal Fusion Report ({args.tag})",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- n_val: `{summary['n_val']}`",
        f"- n_test: `{summary['n_test']}`",
        f"- n_features: `{summary['n_features']}`",
        "",
        "## Best Candidate by Validation QWK",
        "",
        f"- candidate: `{best['candidate']}`",
        f"- val_qwk: `{best['val_qwk']:.6f}`",
        f"- test_qwk: `{best['test_qwk']:.6f}`",
        f"- test_accuracy: `{best['test_accuracy']:.6f}`",
        f"- test_macro_f1: `{best['test_macro_f1']:.6f}`",
        "",
        "## Artifacts",
        "",
        f"- `{csv_path}`",
        f"- `{json_path}`",
    ]
    md_path = run_out / "pass8_internal_fusion_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Best by val: {best['candidate']} | val_qwk={best['val_qwk']:.6f} | test_qwk={best['test_qwk']:.6f}")


if __name__ == "__main__":
    main()
