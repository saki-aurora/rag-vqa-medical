#!/usr/bin/env python3
"""Part 3: Build Chapter 5 figure data and plots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent
OUT_DIR = WORKSPACE_DIR / "out"
DATA_DIR = WORKSPACE_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FREEZE_MANIFEST_PATH = OUT_DIR / "freeze_manifest.json"
PART3_SUMMARY_PATH = OUT_DIR / "part3_ch5_summary.md"


FIELD_DISPLAY = {
    "population": "Population",
    "intervention": "Intervention",
    "comparator": "Comparator",
    "outcomes": "Outcomes",
    "severity_anchors": "Severity anchors",
    "timeframe": "Timeframe",
    "setting": "Setting",
    "constraints": "Constraints",
}


RETRIEVAL_METRIC_SPECS = [
    ("precision_at_k", "Precision@k", "#4E79A7"),
    ("recall_at_k", "Recall@k", "#E15759"),
    ("hit_rate_at_k", "Hit rate@k", "#59A14F"),
]


ABLATION_BACKEND_COLOR = {
    "tfidf": "#4E79A7",
    "hybrid": "#E15759",
    "keyword": "#59A14F",
}


ANSWER_KPI_SPECS = [
    (
        "citation_coverage",
        "Citation coverage",
        True,
        "Citation coverage",
    ),
    (
        "citation_correctness_heuristic",
        "Citation correctness",
        True,
        "Citation correctness",
    ),
    (
        "claim_support_rate_strict",
        "Strict claim support",
        True,
        "Strict claim support",
    ),
    (
        "citation_link_integrity",
        "Citation link integrity",
        True,
        "Citation link integrity",
    ),
    (
        "hallucination_rate_proxy",
        "Hallucination proxy",
        False,
        "Non-hallucination (1-H)",
    ),
    (
        "contradiction_rate_proxy",
        "Contradiction proxy",
        False,
        "Non-contradiction (1-C)",
    ),
    (
        "refusal_rate",
        "Refusal rate",
        False,
        "Answer issuance (1-R)",
    ),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_freeze_index() -> Dict[str, Path]:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing freeze manifest: {FREEZE_MANIFEST_PATH}. Run freeze_inputs.py first."
        )
    manifest = _read_json(FREEZE_MANIFEST_PATH)
    by_id: Dict[str, Path] = {}
    for item in manifest.get("inputs", []):
        input_id = str(item.get("input_id", ""))
        abs_path = Path(str(item.get("abs_path", "")))
        exists = bool(item.get("exists", False))
        if exists:
            by_id[input_id] = abs_path
    return by_id


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_pico_tables(input_index: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    pico_eval = _read_json(input_index["in_ch5_pico_eval"])
    per_query_rows = _read_jsonl(input_index["in_ch5_pico_per_query"])

    fields_obj = pico_eval.get("fields", {})
    field_order = list(pico_eval.get("field_order", list(fields_obj.keys())))
    required_fields = set(str(x) for x in pico_eval.get("required_fields", []))

    aggregate_rows: List[Dict[str, object]] = []
    for field in field_order:
        rec = fields_obj.get(field, {})
        aggregate_rows.append(
            {
                "field": field,
                "field_display": FIELD_DISPLAY.get(field, field.replace("_", " ").title()),
                "is_required": bool(field in required_fields),
                "tp": _as_int(rec.get("tp")),
                "fp": _as_int(rec.get("fp")),
                "fn": _as_int(rec.get("fn")),
                "precision": _as_float(rec.get("precision")),
                "recall": _as_float(rec.get("recall")),
                "f1": _as_float(rec.get("f1")),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)

    per_query_metric_rows: List[Dict[str, object]] = []
    for row in per_query_rows:
        qid = str(row.get("qid", "")).strip()
        query = str(row.get("query", "")).strip()
        fields = row.get("fields", {})
        if not isinstance(fields, dict):
            continue
        for field in field_order:
            stats = fields.get(field, {})
            if not isinstance(stats, dict):
                stats = {}
            tp = _as_int(stats.get("tp"))
            fp = _as_int(stats.get("fp"))
            fn = _as_int(stats.get("fn"))
            precision_q, recall_q, f1_q = _compute_prf(tp, fp, fn)
            per_query_metric_rows.append(
                {
                    "qid": qid,
                    "query": query,
                    "field": field,
                    "field_display": FIELD_DISPLAY.get(field, field.replace("_", " ").title()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision_q": precision_q,
                    "recall_q": recall_q,
                    "f1_q": f1_q,
                }
            )
    per_query_df = pd.DataFrame(per_query_metric_rows)

    stat_rows: List[Dict[str, object]] = []
    for field in field_order:
        sub = per_query_df[per_query_df["field"] == field]
        rec: Dict[str, object] = {"field": field}
        for metric in ["precision_q", "recall_q", "f1_q"]:
            arr = sub[metric].to_numpy(dtype=float) if not sub.empty else np.asarray([], dtype=float)
            if arr.size == 0:
                rec[f"{metric}_mean"] = np.nan
                rec[f"{metric}_ci_low"] = np.nan
                rec[f"{metric}_ci_high"] = np.nan
            else:
                rec[f"{metric}_mean"] = float(np.mean(arr))
                rec[f"{metric}_ci_low"] = float(np.percentile(arr, 2.5))
                rec[f"{metric}_ci_high"] = float(np.percentile(arr, 97.5))
        stat_rows.append(rec)
    per_query_stat_df = pd.DataFrame(stat_rows)

    merged_df = aggregate_df.merge(per_query_stat_df, on="field", how="left")
    merged_df["n_queries"] = _as_int(pico_eval.get("n_queries"))
    merged_df["macro_f1_required_fields"] = _as_float(
        pico_eval.get("macro_f1_required_fields"), default=np.nan
    )
    merged_df["macro_f1_all_fields"] = _as_float(
        pico_eval.get("macro_f1_all_fields"), default=np.nan
    )

    merged_df.to_csv(DATA_DIR / "ch5_pico_field_metrics.csv", index=False)
    per_query_df.to_csv(DATA_DIR / "ch5_pico_per_query_scores.csv", index=False)

    return {
        "aggregate": merged_df,
        "per_query": per_query_df,
    }


def plot_pico_grouped_bar(df: pd.DataFrame) -> Path:
    df = df.copy()
    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, metric in enumerate(["precision", "recall", "f1"]):
        color = {"precision": "#4E79A7", "recall": "#E15759", "f1": "#59A14F"}[metric]
        label = {"precision": "Precision", "recall": "Recall", "f1": "F1"}[metric]
        ax.bar(
            x + (idx - 1) * width,
            df[metric].astype(float).to_numpy(),
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            zorder=3,
        )

    for idx, is_required in enumerate(df["is_required"].tolist()):
        if bool(is_required):
            ax.axvspan(idx - 0.5, idx + 0.5, color="#EFEFEF", alpha=0.35, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(df["field_display"], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Chapter 5 PICO Field Precision/Recall/F1")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.legend(loc="upper right")

    macro_required = _as_float(df["macro_f1_required_fields"].iloc[0], default=np.nan)
    if not np.isnan(macro_required):
        ax.text(
            0.01,
            -0.22,
            f"Required-field macro-F1: {macro_required:.4f}. Shaded categories are required fields.",
            transform=ax.transAxes,
            fontsize=9,
        )

    fig.tight_layout()
    out_path = FIG_DIR / "F07_ch5_pico_field_precision_recall_f1.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_retrieval_curve_table(input_index: Dict[str, Path]) -> pd.DataFrame:
    retrieval_eval = _read_json(input_index["in_ch5_retrieval_eval"])
    metrics_obj = retrieval_eval.get("metrics", {})
    ci_obj = retrieval_eval.get("metrics_bootstrap_ci", {})
    k_values = [int(k) for k in retrieval_eval.get("k_values", [])]

    rows: List[Dict[str, object]] = []
    for k in k_values:
        k_key = f"k={k}"
        rec = metrics_obj.get(k_key, {})
        ci_rec = ci_obj.get(k_key, {})
        for metric_key, metric_display, _ in RETRIEVAL_METRIC_SPECS:
            rows.append(
                {
                    "k": k,
                    "metric_key": metric_key,
                    "metric_display": metric_display,
                    "value": _as_float(rec.get(metric_key), default=np.nan),
                    "ci_low": _as_float(ci_rec.get(f"{metric_key}_ci_low"), default=np.nan),
                    "ci_high": _as_float(ci_rec.get(f"{metric_key}_ci_high"), default=np.nan),
                    "n_queries": _as_int(retrieval_eval.get("n_queries")),
                }
            )
    df = pd.DataFrame(rows).sort_values(["metric_key", "k"]).reset_index(drop=True)
    df.to_csv(DATA_DIR / "ch5_retrieval_curve.csv", index=False)
    return df


def plot_retrieval_curve(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric_key, metric_display, color in RETRIEVAL_METRIC_SPECS:
        sub = df[df["metric_key"] == metric_key].sort_values("k")
        if sub.empty:
            continue
        x = sub["k"].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, color=color, label=metric_display)
        if not np.isnan(low).all() and not np.isnan(high).all():
            ax.fill_between(x, low, high, color=color, alpha=0.15)

    unique_k = sorted(df["k"].unique().tolist())
    ax.set_xticks(unique_k)
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Chapter 5 Retrieval@k with 95% Bootstrap CIs")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = FIG_DIR / "F08_ch5_retrieval_at_k_curve_with_ci.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _ablation_display_label(row: pd.Series) -> str:
    backend = str(row.get("backend", "")).strip().upper()
    rerank = bool(row.get("rerank", False))
    alpha = _as_float(row.get("alpha"), default=np.nan)
    if rerank:
        if np.isnan(alpha):
            return f"{backend} + rerank"
        return f"{backend} + rerank a={alpha:.2f}"
    return f"{backend} no rerank"


def build_ablation_table(input_index: Dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(input_index["in_ch5_retrieval_ablation"], sep="\t")
    for col in ["alpha", "hit@1", "hit@3", "hit@5", "p@1", "p@5", "r@5"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rerank" in df.columns:
        df["rerank"] = df["rerank"].map(lambda v: str(v).strip().lower() == "true")
    else:
        df["rerank"] = False
    df["method_display"] = df.apply(_ablation_display_label, axis=1)
    df["backend"] = df["backend"].astype(str).str.strip().str.lower()
    df = df.sort_values(["r@5", "hit@5", "hit@1"], ascending=[True, True, True]).reset_index(drop=True)
    df.to_csv(DATA_DIR / "ch5_retrieval_ablation.csv", index=False)
    return df


def plot_ablation_lollipop(df: pd.DataFrame) -> Path:
    vals = df["r@5"].to_numpy(dtype=float)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, row in df.iterrows():
        val = _as_float(row.get("r@5"), default=np.nan)
        backend = str(row.get("backend", "")).strip().lower()
        color = ABLATION_BACKEND_COLOR.get(backend, "#B07AA1")
        marker = "o" if bool(row.get("rerank", False)) else "s"
        ax.hlines(y=idx, xmin=0, xmax=val, color=color, linewidth=2, alpha=0.9, zorder=2)
        ax.scatter(
            val,
            idx,
            s=90,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        ax.text(min(0.98, val + 0.012), idx, f"{val:.2f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(df["method_display"])
    x_max = max(0.5, float(np.nanmax(vals)) + 0.08)
    ax.set_xlim(0, min(1.0, x_max))
    ax.set_xlabel("Recall@5 (higher is better)")
    ax.set_title("Chapter 5 Retrieval Ablation Comparison (Recall@5)")
    ax.grid(axis="x", alpha=0.25)

    backend_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=backend.upper())
        for backend, color in ABLATION_BACKEND_COLOR.items()
        if backend in set(df["backend"].tolist())
    ]
    rerank_handles = [
        plt.Line2D([0], [0], marker="o", color="black", lw=0, label="Rerank on"),
        plt.Line2D([0], [0], marker="s", color="black", lw=0, label="Rerank off"),
    ]
    ax.legend(
        handles=backend_handles + rerank_handles,
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    out_path = FIG_DIR / "F09_ch5_retrieval_ablation_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _checklist_label(key: str) -> str:
    mapping = {
        "kb_index_built": "KB index built",
        "wrapper_ran_min_queries": "Wrapper ran minimum queries",
        "pico_eval_exists": "PICO eval exists",
        "retrieval_eval_exists": "Retrieval eval exists",
        "answer_eval_exists": "Answer eval exists",
        "chapter_text_synced": "Chapter text synced",
    }
    return mapping.get(key, key.replace("_", " ").title())


def build_answer_kpi_tables(input_index: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    answer_eval = _read_json(input_index["in_ch5_answer_eval"])
    completion_report = _read_json(input_index["in_ch5_completion_audit_pass4"])

    n_outputs = _as_int(answer_eval.get("n_outputs"))
    refusal_count = _as_int(answer_eval.get("refusal_count"))
    refusal_rate = _safe_div(refusal_count, n_outputs)

    kpi_source = dict(answer_eval)
    kpi_source["refusal_rate"] = refusal_rate

    kpi_rows: List[Dict[str, object]] = []
    for key, label, higher_is_better, panel_label in ANSWER_KPI_SPECS:
        value = _as_float(kpi_source.get(key), default=np.nan)
        panel_value = value if higher_is_better else (1.0 - value if not np.isnan(value) else np.nan)
        kpi_rows.append(
            {
                "kpi_key": key,
                "kpi_label": label,
                "value": value,
                "higher_is_better": bool(higher_is_better),
                "panel_label": panel_label,
                "panel_value": panel_value,
            }
        )
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(DATA_DIR / "ch5_answer_kpis.csv", index=False)

    checklist = completion_report.get("checklist", {})
    checklist_rows: List[Dict[str, object]] = []
    if isinstance(checklist, dict):
        for key, value in checklist.items():
            checklist_rows.append(
                {
                    "check_key": key,
                    "check_label": _checklist_label(key),
                    "status_bool": bool(value),
                    "status_text": "PASS" if bool(value) else "FAIL",
                }
            )
    checklist_df = pd.DataFrame(checklist_rows)
    checklist_df.to_csv(DATA_DIR / "ch5_completion_audit_checklist.csv", index=False)

    counts_df = pd.DataFrame(
        [
            {
                "audit_status": str(completion_report.get("status", "UNKNOWN")).upper(),
                "n_outputs": n_outputs,
                "n_claims": _as_int(answer_eval.get("n_claims")),
                "n_claims_evaluated": _as_int(answer_eval.get("n_claims_evaluated")),
                "n_policy_claims_excluded": _as_int(answer_eval.get("n_policy_claims_excluded")),
                "refusal_count": refusal_count,
                "refusal_rate": refusal_rate,
            }
        ]
    )
    counts_df.to_csv(DATA_DIR / "ch5_answer_counts.csv", index=False)

    return {
        "kpis": kpi_df,
        "checklist": checklist_df,
        "counts": counts_df,
    }


def plot_answer_kpi_panel(
    kpi_df: pd.DataFrame, checklist_df: pd.DataFrame, counts_df: pd.DataFrame
) -> Path:
    counts = counts_df.iloc[0].to_dict()
    audit_status = str(counts.get("audit_status", "UNKNOWN")).upper()
    refusal_rate = _as_float(counts.get("refusal_rate"), default=np.nan)
    refusal_count = _as_int(counts.get("refusal_count"))
    n_outputs = _as_int(counts.get("n_outputs"))
    n_claims_evaluated = _as_int(counts.get("n_claims_evaluated"))

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 3.1], width_ratios=[1.8, 1.0])
    ax_header = fig.add_subplot(grid[0, :])
    ax_bar = fig.add_subplot(grid[1, 0])
    ax_audit = fig.add_subplot(grid[1, 1])

    ax_header.axis("off")
    ax_header.text(
        0.01,
        0.66,
        "Chapter 5 Answer Quality and Grounding KPI Panel",
        fontsize=14,
        fontweight="bold",
    )
    ax_header.text(
        0.01,
        0.22,
        (
            f"Outputs={n_outputs} | Claims evaluated={n_claims_evaluated} | "
            f"Refusals={refusal_count} ({refusal_rate:.1%}) | Audit={audit_status}"
        ),
        fontsize=11,
    )

    plot_df = kpi_df.copy().sort_values("panel_value", ascending=True).reset_index(drop=True)
    ypos = np.arange(len(plot_df))
    bars = ax_bar.barh(
        ypos,
        plot_df["panel_value"].to_numpy(dtype=float),
        color="#4E79A7",
        edgecolor="black",
        linewidth=0.3,
    )
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(plot_df["panel_label"])
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Normalized quality score")
    ax_bar.set_title("Answer quality / grounding scores")
    ax_bar.grid(axis="x", alpha=0.25)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        raw = _as_float(row["value"], default=np.nan)
        ax_bar.text(
            min(0.98, float(row["panel_value"]) + 0.015),
            idx,
            f"raw={raw:.3f}",
            va="center",
            fontsize=9,
        )

    ax_audit.set_title("Completion audit checklist")
    ax_audit.set_xlim(0, 1)
    ax_audit.set_ylim(0, 1)
    ax_audit.axis("off")

    if checklist_df.empty:
        ax_audit.text(0.02, 0.9, "No checklist rows found", fontsize=10)
    else:
        n_rows = len(checklist_df)
        step = 0.9 / max(1, n_rows)
        for idx, (_, row) in enumerate(checklist_df.iterrows()):
            y = 0.95 - idx * step
            status = str(row["status_text"])
            color = "#59A14F" if status == "PASS" else "#E15759"
            ax_audit.text(0.02, y, str(row["check_label"]), fontsize=10, va="center")
            ax_audit.text(
                0.98,
                y,
                status,
                fontsize=10,
                va="center",
                ha="right",
                color=color,
                fontweight="bold",
            )
            ax_audit.hlines(y - (step * 0.4), 0.02, 0.98, color="#DDDDDD", linewidth=0.8)

    fig.tight_layout()
    out_path = FIG_DIR / "F10_ch5_answer_quality_grounding_kpi_panel.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_part3_summary(outputs: Dict[str, Path]) -> None:
    lines = [
        "# Part 3 (Chapter 5 Figures) Summary",
        "",
        f"- Generated UTC: `{_utc_now()}`",
        "",
        "## Data Outputs",
    ]
    for key in [
        "ch5_pico_field_metrics_csv",
        "ch5_pico_per_query_scores_csv",
        "ch5_retrieval_curve_csv",
        "ch5_retrieval_ablation_csv",
        "ch5_answer_kpis_csv",
        "ch5_completion_audit_checklist_csv",
        "ch5_answer_counts_csv",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.extend(["", "## Figure Outputs"])
    for key in [
        "f07_pico_bar_png",
        "f08_retrieval_curve_png",
        "f09_ablation_png",
        "f10_answer_kpi_panel_png",
    ]:
        lines.append(f"- `{key}`: `{outputs[key]}`")
    PART3_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    input_index = _load_freeze_index()

    pico_tables = build_pico_tables(input_index)
    f07_png = plot_pico_grouped_bar(pico_tables["aggregate"])

    retrieval_curve_df = build_retrieval_curve_table(input_index)
    f08_png = plot_retrieval_curve(retrieval_curve_df)

    ablation_df = build_ablation_table(input_index)
    f09_png = plot_ablation_lollipop(ablation_df)

    answer_tables = build_answer_kpi_tables(input_index)
    f10_png = plot_answer_kpi_panel(
        answer_tables["kpis"],
        answer_tables["checklist"],
        answer_tables["counts"],
    )

    outputs = {
        "ch5_pico_field_metrics_csv": DATA_DIR / "ch5_pico_field_metrics.csv",
        "ch5_pico_per_query_scores_csv": DATA_DIR / "ch5_pico_per_query_scores.csv",
        "ch5_retrieval_curve_csv": DATA_DIR / "ch5_retrieval_curve.csv",
        "ch5_retrieval_ablation_csv": DATA_DIR / "ch5_retrieval_ablation.csv",
        "ch5_answer_kpis_csv": DATA_DIR / "ch5_answer_kpis.csv",
        "ch5_completion_audit_checklist_csv": DATA_DIR / "ch5_completion_audit_checklist.csv",
        "ch5_answer_counts_csv": DATA_DIR / "ch5_answer_counts.csv",
        "f07_pico_bar_png": f07_png,
        "f08_retrieval_curve_png": f08_png,
        "f09_ablation_png": f09_png,
        "f10_answer_kpi_panel_png": f10_png,
    }
    write_part3_summary(outputs)

    print("[part3] Chapter 5 figure pipeline completed")
    for key, path in outputs.items():
        print(f"[part3] {key}: {path}")
    print(f"[part3] summary: {PART3_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
