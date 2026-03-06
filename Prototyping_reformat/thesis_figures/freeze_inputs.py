#!/usr/bin/env python3
"""Freeze canonical thesis-figure inputs and emit lock manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


WORKSPACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE_DIR.parents[1]
OUT_DIR = WORKSPACE_DIR / "out"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CANONICAL_INPUTS: List[Dict[str, object]] = [
    # Chapter 3 synthesis sources
    {
        "input_id": "in_ch3_hyperkvasir_report",
        "group": "ch3_reports",
        "path": "Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md",
        "required": True,
        "description": "Cross-dataset synthesis source report",
    },
    {
        "input_id": "in_ch3_imageclef_report",
        "group": "ch3_reports",
        "path": "Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md",
        "required": True,
        "description": "Cross-dataset synthesis source report",
    },
    {
        "input_id": "in_ch3_kvasir_vqa_report",
        "group": "ch3_reports",
        "path": "Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md",
        "required": True,
        "description": "Cross-dataset synthesis source report",
    },
    {
        "input_id": "in_ch3_kvasir_vqa_x1_report",
        "group": "ch3_reports",
        "path": "Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md",
        "required": True,
        "description": "Cross-dataset synthesis source report",
    },
    {
        "input_id": "in_ch3_limuc_report",
        "group": "ch3_reports",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md",
        "required": True,
        "description": "Cross-dataset synthesis source report",
    },
    {
        "input_id": "in_ch3_kvasir_seg_report",
        "group": "ch3_reports",
        "path": "Prototyping_reformat/DatasetAnalysis/Kvasir_SEG/Kvasir_SEG.md",
        "required": True,
        "description": "Cross-dataset synthesis source report",
    },
    # Chapter 4 metrics (run-level)
    {
        "input_id": "in_ch4_metrics_clip_linear",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/results/clip_linear_baseline/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics",
    },
    {
        "input_id": "in_ch4_metrics_resnet50_frozen",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/results/resnet50_frozen_logreg/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics",
    },
    {
        "input_id": "in_ch4_metrics_vit_frozen",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/results/vit_frozen_logreg/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics",
    },
    {
        "input_id": "in_ch4_metrics_finetune_resnet50",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_resnet50/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics",
    },
    {
        "input_id": "in_ch4_metrics_finetune_vit_or_swin",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_vit_or_swin/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics",
    },
    {
        "input_id": "in_ch4_metrics_vlm_zero_shot",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_zero_shot_mayo/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics",
    },
    {
        "input_id": "in_ch4_metrics_vlm_zero_shot_mode2_sampling",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_zero_shot_mode2_label_sampling_full_20260302/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics (controlled decoding)",
    },
    {
        "input_id": "in_ch4_metrics_vlm_lora_balanced_full",
        "group": "ch4_metrics",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_finetune_mayo_balanced_full_20260303/metrics_test.json",
        "required": True,
        "description": "Chapter 4 run-level metrics (best generative run)",
    },
    # Chapter 4 pred_test for remission/significance calculations
    {
        "input_id": "in_ch4_pred_clip_linear",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/results/clip_linear_baseline/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_resnet50_frozen",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/results/resnet50_frozen_logreg/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_vit_frozen",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/results/vit_frozen_logreg/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_finetune_resnet50",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_resnet50/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_finetune_vit_or_swin",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_vit_or_swin/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_vlm_zero_shot",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_zero_shot_mayo/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_vlm_zero_shot_mode2_sampling",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_zero_shot_mode2_label_sampling_full_20260302/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    {
        "input_id": "in_ch4_pred_vlm_lora_balanced_full",
        "group": "ch4_predictions",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_finetune_mayo_balanced_full_20260303/pred_test.csv",
        "required": True,
        "description": "Predictions for remission and paired significance visuals",
    },
    # Chapter 4 confusion matrix figure panel
    {
        "input_id": "in_ch4_confusion_best_supervised",
        "group": "ch4_confusion",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_resnet50/confusion_test.png",
        "required": True,
        "description": "Best supervised confusion matrix",
    },
    {
        "input_id": "in_ch4_confusion_best_generative",
        "group": "ch4_confusion",
        "path": "Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_finetune_mayo_balanced_full_20260303/confusion_test.png",
        "required": True,
        "description": "Best generative confusion matrix",
    },
    # Chapter 5 pass4_latest canonical inputs
    {
        "input_id": "in_ch5_pico_eval",
        "group": "ch5_eval",
        "path": "Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json",
        "required": True,
        "description": "PICO aggregate evaluation",
    },
    {
        "input_id": "in_ch5_pico_per_query",
        "group": "ch5_eval",
        "path": "Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval_per_query.jsonl",
        "required": True,
        "description": "PICO per-query evaluation",
    },
    {
        "input_id": "in_ch5_retrieval_eval",
        "group": "ch5_eval",
        "path": "Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json",
        "required": True,
        "description": "Retrieval@k plus bootstrap intervals",
    },
    {
        "input_id": "in_ch5_answer_eval",
        "group": "ch5_eval",
        "path": "Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json",
        "required": True,
        "description": "Answer quality and grounding evaluation",
    },
    {
        "input_id": "in_ch5_retrieval_ablation",
        "group": "ch5_eval",
        "path": "Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass3_ablation/retrieval_ablation_summary.tsv",
        "required": True,
        "description": "Retrieval backend/rerank ablation summary",
    },
    {
        "input_id": "in_ch5_completion_audit_pass4",
        "group": "ch5_eval",
        "path": "Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_pass4_latest/chapter5_completion_report.json",
        "required": True,
        "description": "Completion audit and PASS status for pass4_latest",
    },
]


FIGURES: List[Dict[str, object]] = [
    {
        "figure_id": "F01",
        "chapter": "3",
        "title": "Cross-dataset benchmark heatmap",
        "plot_type": "heatmap",
        "source_input_ids": [
            "in_ch3_hyperkvasir_report",
            "in_ch3_imageclef_report",
            "in_ch3_kvasir_vqa_report",
            "in_ch3_kvasir_vqa_x1_report",
            "in_ch3_limuc_report",
            "in_ch3_kvasir_seg_report",
        ],
    },
    {
        "figure_id": "F02",
        "chapter": "4",
        "title": "Chapter 4 core metric comparison",
        "plot_type": "grouped_bar",
        "source_input_ids": [
            "in_ch4_metrics_clip_linear",
            "in_ch4_metrics_resnet50_frozen",
            "in_ch4_metrics_vit_frozen",
            "in_ch4_metrics_finetune_resnet50",
            "in_ch4_metrics_finetune_vit_or_swin",
            "in_ch4_metrics_vlm_zero_shot",
            "in_ch4_metrics_vlm_zero_shot_mode2_sampling",
            "in_ch4_metrics_vlm_lora_balanced_full",
        ],
    },
    {
        "figure_id": "F03",
        "chapter": "4",
        "title": "Chapter 4 radar profile",
        "plot_type": "radar",
        "source_input_ids": [
            "in_ch4_metrics_finetune_resnet50",
            "in_ch4_metrics_vit_frozen",
            "in_ch4_metrics_vlm_zero_shot_mode2_sampling",
            "in_ch4_metrics_vlm_lora_balanced_full",
        ],
    },
    {
        "figure_id": "F04",
        "chapter": "4",
        "title": "Remission slice comparison",
        "plot_type": "grouped_bar",
        "source_input_ids": [
            "in_ch4_pred_clip_linear",
            "in_ch4_pred_resnet50_frozen",
            "in_ch4_pred_vit_frozen",
            "in_ch4_pred_finetune_resnet50",
            "in_ch4_pred_finetune_vit_or_swin",
            "in_ch4_pred_vlm_zero_shot",
            "in_ch4_pred_vlm_zero_shot_mode2_sampling",
            "in_ch4_pred_vlm_lora_balanced_full",
        ],
    },
    {
        "figure_id": "F05",
        "chapter": "4",
        "title": "Paired McNemar significance heatmap",
        "plot_type": "heatmap",
        "source_input_ids": [
            "in_ch4_pred_clip_linear",
            "in_ch4_pred_resnet50_frozen",
            "in_ch4_pred_vit_frozen",
            "in_ch4_pred_finetune_resnet50",
            "in_ch4_pred_finetune_vit_or_swin",
            "in_ch4_pred_vlm_zero_shot",
            "in_ch4_pred_vlm_zero_shot_mode2_sampling",
            "in_ch4_pred_vlm_lora_balanced_full",
        ],
    },
    {
        "figure_id": "F06",
        "chapter": "4",
        "title": "Best supervised vs best generative confusion panel",
        "plot_type": "panel",
        "source_input_ids": [
            "in_ch4_confusion_best_supervised",
            "in_ch4_confusion_best_generative",
        ],
    },
    {
        "figure_id": "F07",
        "chapter": "5",
        "title": "PICO field precision/recall/F1",
        "plot_type": "grouped_bar",
        "source_input_ids": [
            "in_ch5_pico_eval",
            "in_ch5_pico_per_query",
        ],
    },
    {
        "figure_id": "F08",
        "chapter": "5",
        "title": "Retrieval@k curve with confidence intervals",
        "plot_type": "line_with_ci",
        "source_input_ids": [
            "in_ch5_retrieval_eval",
        ],
    },
    {
        "figure_id": "F09",
        "chapter": "5",
        "title": "Retrieval ablation comparison",
        "plot_type": "lollipop",
        "source_input_ids": [
            "in_ch5_retrieval_ablation",
        ],
    },
    {
        "figure_id": "F10",
        "chapter": "5",
        "title": "Answer quality and grounding KPI panel",
        "plot_type": "kpi_panel",
        "source_input_ids": [
            "in_ch5_answer_eval",
            "in_ch5_completion_audit_pass4",
        ],
    },
]


def _input_record(entry: Dict[str, object]) -> Dict[str, object]:
    rel_path = Path(str(entry["path"]))
    abs_path = REPO_ROOT / rel_path
    exists = abs_path.exists()
    record: Dict[str, object] = {
        "input_id": entry["input_id"],
        "group": entry["group"],
        "description": entry["description"],
        "required": bool(entry["required"]),
        "path": str(rel_path),
        "abs_path": str(abs_path),
        "exists": exists,
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
    }
    if exists:
        stat = abs_path.stat()
        record["size_bytes"] = int(stat.st_size)
        record["mtime_utc"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).replace(microsecond=0).isoformat()
        record["sha256"] = _sha256_file(abs_path)
    return record


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


def _build_figure_rows(
    input_index: Dict[str, Dict[str, object]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for fig in FIGURES:
        source_ids = [str(x) for x in fig["source_input_ids"]]
        missing = [sid for sid in source_ids if not bool(input_index[sid]["exists"])]
        rows.append(
            {
                "figure_id": fig["figure_id"],
                "chapter": fig["chapter"],
                "title": fig["title"],
                "plot_type": fig["plot_type"],
                "source_input_ids": ";".join(source_ids),
                "status": "ready" if not missing else "blocked",
                "missing_input_ids": ";".join(missing),
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_rows = [_input_record(entry) for entry in CANONICAL_INPUTS]
    input_index = {str(row["input_id"]): row for row in input_rows}

    missing_required = [
        str(row["input_id"])
        for row in input_rows
        if bool(row["required"]) and not bool(row["exists"])
    ]

    freeze_manifest = {
        "generated_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "workspace_dir": str(WORKSPACE_DIR),
        "status": "PASS" if not missing_required else "FAIL",
        "total_inputs": len(input_rows),
        "missing_required_count": len(missing_required),
        "missing_required_input_ids": missing_required,
        "inputs": input_rows,
    }
    (OUT_DIR / "freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2), encoding="utf-8"
    )
    _write_csv(input_rows, OUT_DIR / "freeze_manifest.csv")

    figure_rows = _build_figure_rows(input_index)
    _write_csv(figure_rows, OUT_DIR / "figure_manifest.csv")

    ready_count = sum(1 for row in figure_rows if row["status"] == "ready")
    print(f"[freeze] status={freeze_manifest['status']}")
    print(f"[freeze] inputs={len(input_rows)} required_missing={len(missing_required)}")
    print(f"[freeze] figures_ready={ready_count}/{len(figure_rows)}")
    print(f"[freeze] wrote: {OUT_DIR / 'freeze_manifest.json'}")
    print(f"[freeze] wrote: {OUT_DIR / 'freeze_manifest.csv'}")
    print(f"[freeze] wrote: {OUT_DIR / 'figure_manifest.csv'}")

    return 0 if not missing_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
