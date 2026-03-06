#!/usr/bin/env python3
"""Part 5: Stage generated figures and emit thesis-integration snippets."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


WORKSPACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE_DIR.parents[1]
OUT_DIR = WORKSPACE_DIR / "out"
DATA_DIR = WORKSPACE_DIR / "data"
FIG_OUT_DIR = OUT_DIR / "figures"
THESIS_MD_DIR = REPO_ROOT / "Thesis" / "markdown"
THESIS_FIG_DIR = THESIS_MD_DIR / "figures" / "generated"

FIGURE_MANIFEST_PATH = OUT_DIR / "figure_manifest.csv"
PART5_SUMMARY_PATH = OUT_DIR / "part5_thesis_integration_summary.md"
PART5_INSERTIONS_PATH = OUT_DIR / "thesis_figure_insertions.md"


GENERATED_FIGURES = {
    "F01": "F01_ch3_cross_dataset_benchmark_heatmap.png",
    "F02": "F02_ch4_core_metric_comparison.png",
    "F03": "F03_ch4_radar_profile.png",
    "F04": "F04_ch4_remission_slice_comparison.png",
    "F05": "F05_ch4_mcnemar_significance_heatmap.png",
    "F06": "F06_ch4_confusion_panel.png",
    "F07": "F07_ch5_pico_field_precision_recall_f1.png",
    "F08": "F08_ch5_retrieval_at_k_curve_with_ci.png",
    "F09": "F09_ch5_retrieval_ablation_comparison.png",
    "F10": "F10_ch5_answer_quality_grounding_kpi_panel.png",
}


CAPTIONS = {
    "F01": "Cross-dataset reliability benchmark heatmap using frozen Chapter 3 report artifacts.",
    "F02": "Chapter 4 core metric comparison across frozen, supervised, and generative model families.",
    "F03": "Normalized radar profile for selected Chapter 4 representative models.",
    "F04": "Remission-slice comparison (Mayo 0-1 vs 2-3) across Chapter 4 runs.",
    "F05": "Pairwise McNemar significance heatmap for Chapter 4 model comparisons.",
    "F06": "Confusion-matrix panel: best supervised run versus best generative run (Chapter 4).",
    "F07": "PICO field precision, recall, and F1 from Chapter 5 evaluation artifacts.",
    "F08": "Retrieval@k curve with 95% bootstrap confidence intervals (Chapter 5).",
    "F09": "Retrieval ablation lollipop comparison across backend and rerank settings.",
    "F10": "Answer-quality and grounding KPI panel with completion-audit checklist (Chapter 5).",
}


CHAPTER_FILE_BY_NUM = {
    "3": "03_chapter_3_investigating_existing_vqa_techniques_across_gi_endoscopy_datasets_v2.md",
    "4": "04_chapter_4_developing_the_proposed_approach.md",
    "5": "05_chapter_5_genai_wrapper_pico.md",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_figure_manifest() -> List[Dict[str, str]]:
    if not FIGURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing figure manifest: {FIGURE_MANIFEST_PATH}")
    rows: List[Dict[str, str]] = []
    with FIGURE_MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: str(v) for k, v in row.items()})
    return rows


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stage_figures(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    staged_rows: List[Dict[str, object]] = []

    for row in manifest_rows:
        figure_id = row["figure_id"]
        chapter = row["chapter"]
        title = row["title"]
        file_name = GENERATED_FIGURES.get(figure_id, "")
        src_path = FIG_OUT_DIR / file_name if file_name else Path()
        src_exists = bool(file_name) and src_path.exists()
        if src_exists:
            dst_path = THESIS_FIG_DIR / file_name
            shutil.copy2(src_path, dst_path)
            staged_rel = f"figures/generated/{file_name}"
        else:
            dst_path = Path()
            staged_rel = ""

        chapter_file = CHAPTER_FILE_BY_NUM.get(chapter, "")
        chapter_abs = THESIS_MD_DIR / chapter_file if chapter_file else Path()
        chapter_exists = bool(chapter_file) and chapter_abs.exists()
        caption = CAPTIONS.get(figure_id, title)
        md_snippet = (
            f"![{title} ({figure_id})]({staged_rel})\n\n"
            f"*{caption}*"
            if staged_rel
            else ""
        )

        staged_rows.append(
            {
                "figure_id": figure_id,
                "chapter": chapter,
                "title": title,
                "plot_type": row["plot_type"],
                "manifest_status": row["status"],
                "source_input_ids": row["source_input_ids"],
                "generated_file_name": file_name,
                "generated_abs_path": str(src_path) if file_name else "",
                "generated_exists": src_exists,
                "staged_abs_path": str(dst_path) if src_exists else "",
                "staged_relative_path_from_chapter_md": staged_rel,
                "caption": caption,
                "chapter_md_file": chapter_file,
                "chapter_md_abs_path": str(chapter_abs) if chapter_file else "",
                "chapter_md_exists": chapter_exists,
                "markdown_snippet": md_snippet,
            }
        )

    return staged_rows


def write_insertion_guide(staged_rows: List[Dict[str, object]]) -> None:
    by_chapter: Dict[str, List[Dict[str, object]]] = {"3": [], "4": [], "5": []}
    for row in staged_rows:
        ch = str(row["chapter"])
        if ch in by_chapter:
            by_chapter[ch].append(row)

    lines: List[str] = [
        "# Thesis Figure Insertions (Generated)",
        "",
        f"- Generated UTC: `{_utc_now()}`",
        "",
        "Use the snippets below inside the corresponding chapter markdown files.",
        "",
    ]

    for ch in ["3", "4", "5"]:
        chapter_file = CHAPTER_FILE_BY_NUM[ch]
        lines.extend(
            [
                f"## Chapter {ch}",
                "",
                f"Target file: `Thesis/markdown/{chapter_file}`",
                "",
            ]
        )
        chapter_rows = sorted(by_chapter[ch], key=lambda r: r["figure_id"])
        if not chapter_rows:
            lines.append("_No generated figures in this chapter._")
            lines.append("")
            continue

        for row in chapter_rows:
            lines.extend(
                [
                    f"### {row['figure_id']} - {row['title']}",
                    "",
                    "```markdown",
                    str(row["markdown_snippet"]),
                    "```",
                    "",
                ]
            )

    PART5_INSERTIONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(staged_rows: List[Dict[str, object]], catalog_path: Path) -> None:
    total = len(staged_rows)
    generated_ok = sum(1 for r in staged_rows if bool(r["generated_exists"]))
    staged_ok = sum(1 for r in staged_rows if bool(r["staged_abs_path"]))
    chapter_targets = sorted(
        {str(r["chapter_md_file"]) for r in staged_rows if str(r["chapter_md_file"])}
    )

    lines = [
        "# Part 5 (Thesis Integration) Summary",
        "",
        f"- Generated UTC: `{_utc_now()}`",
        f"- Figures in manifest: `{total}`",
        f"- Generated figure files found: `{generated_ok}`",
        f"- Figures staged to thesis markdown directory: `{staged_ok}`",
        "",
        "## Outputs",
        f"- `catalog_csv`: `{catalog_path}`",
        f"- `insertions_md`: `{PART5_INSERTIONS_PATH}`",
        f"- `staged_dir`: `{THESIS_FIG_DIR}`",
        "",
        "## Target Chapter Files",
    ]
    for chapter_file in chapter_targets:
        lines.append(f"- `Thesis/markdown/{chapter_file}`")

    missing = [r for r in staged_rows if not bool(r["generated_exists"])]
    if missing:
        lines.extend(["", "## Missing Generated Figure Files"])
        for row in missing:
            lines.append(
                f"- `{row['figure_id']}` expected `{row['generated_file_name']}` at `{row['generated_abs_path']}`"
            )

    PART5_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_figure_manifest()
    staged_rows = stage_figures(manifest_rows)
    catalog_path = DATA_DIR / "thesis_figure_catalog.csv"
    _write_csv(staged_rows, catalog_path)
    write_insertion_guide(staged_rows)
    write_summary(staged_rows, catalog_path)

    print("[part5] Thesis integration packaging completed")
    print(f"[part5] catalog: {catalog_path}")
    print(f"[part5] insertion guide: {PART5_INSERTIONS_PATH}")
    print(f"[part5] summary: {PART5_SUMMARY_PATH}")
    print(f"[part5] staged_dir: {THESIS_FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
