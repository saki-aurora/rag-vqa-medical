#!/usr/bin/env python3
"""Chapter 5 completion audit for the PICO wrapper workstream."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


CHAPTER4_RUN_ID = "vlm_lora_finetune_mayo_balanced_full_20260303_20260303T080754Z"


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _bool_icon(v: bool) -> str:
    return "✅" if v else "❌"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    workspace_root = _find_workspace_root()
    repo_root = _find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kb_manifest",
        type=Path,
        default=workspace_root / "results" / "kb_build_latest" / "kb_manifest.json",
    )
    parser.add_argument(
        "--wrapper_outputs",
        type=Path,
        default=workspace_root / "results" / "wrapper_eval_latest" / "wrapper_outputs.jsonl",
    )
    parser.add_argument(
        "--run_config",
        type=Path,
        default=workspace_root / "results" / "wrapper_eval_latest" / "run_config.json",
    )
    parser.add_argument(
        "--pico_eval",
        type=Path,
        default=workspace_root / "results" / "eval_latest" / "pico_eval.json",
    )
    parser.add_argument(
        "--retrieval_eval",
        type=Path,
        default=workspace_root / "results" / "eval_latest" / "retrieval_eval.json",
    )
    parser.add_argument(
        "--answer_eval",
        type=Path,
        default=workspace_root / "results" / "eval_latest" / "answer_eval.json",
    )
    parser.add_argument(
        "--chapter_md",
        type=Path,
        default=repo_root / "Thesis" / "markdown" / "05_chapter_5_genai_wrapper_pico.md",
    )
    parser.add_argument("--min_queries", type=int, default=20)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Optional audit output directory. Defaults to results/chapter5_completion_audit_<utc>/",
    )
    parser.add_argument("--audit_run_id", type=str, default=None)
    return parser.parse_args()


def _check_kb(manifest_path: Path) -> Tuple[bool, Dict[str, object], List[str]]:
    notes: List[str] = []
    if not manifest_path.exists():
        return False, {}, [f"Missing KB manifest: {manifest_path}"]
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return False, {}, [f"Failed to parse KB manifest: {exc}"]

    index_file = Path(str(manifest.get("index_file", "")))
    if not index_file.is_absolute():
        index_file = (manifest_path.parent / index_file).resolve()
    n_chunks = int(manifest.get("n_chunks", 0) or 0)
    n_docs = int(manifest.get("n_docs", 0) or 0)

    ok = index_file.exists() and n_chunks > 0 and n_docs > 0
    if not index_file.exists():
        notes.append(f"Index file missing: {index_file}")
    if n_chunks <= 0:
        notes.append("Manifest reports no chunks.")
    if n_docs <= 0:
        notes.append("Manifest reports no documents.")

    details = {
        "manifest_path": str(manifest_path),
        "index_file": str(index_file),
        "index_backend": manifest.get("index_backend"),
        "n_chunks": n_chunks,
        "n_docs": n_docs,
        "n_source_files": int(manifest.get("n_source_files", 0) or 0),
    }
    return ok, details, notes


def _check_wrapper_outputs(
    wrapper_outputs_path: Path,
    run_config_path: Path,
    min_queries: int,
) -> Tuple[bool, Dict[str, object], List[str]]:
    notes: List[str] = []
    if not wrapper_outputs_path.exists():
        return False, {}, [f"Missing wrapper outputs: {wrapper_outputs_path}"]

    try:
        n_outputs = _count_jsonl_rows(wrapper_outputs_path)
    except Exception as exc:
        return False, {}, [f"Failed reading wrapper outputs: {exc}"]

    run_id = None
    if run_config_path.exists():
        try:
            rc = _read_json(run_config_path)
            run_id = rc.get("run_id")
        except Exception:
            run_id = None
    else:
        notes.append(f"run_config missing: {run_config_path}")

    ok = n_outputs >= min_queries
    if not ok:
        notes.append(f"Only {n_outputs} wrapper outputs found (< {min_queries}).")

    details = {
        "wrapper_outputs_path": str(wrapper_outputs_path),
        "run_config_path": str(run_config_path),
        "wrapper_run_id": run_id,
        "n_wrapper_outputs": n_outputs,
        "min_queries_required": min_queries,
    }
    return ok, details, notes


def _check_eval_file(path: Path, label: str) -> Tuple[bool, Dict[str, object], List[str]]:
    notes: List[str] = []
    if not path.exists():
        return False, {"path": str(path), "label": label}, [f"Missing {label}: {path}"]
    try:
        data = _read_json(path)
    except Exception as exc:
        return False, {"path": str(path), "label": label}, [f"Failed to parse {label}: {exc}"]
    details = {
        "path": str(path),
        "label": label,
        "task": data.get("task"),
    }
    return True, details, notes


def _check_chapter_sync(
    chapter_md: Path,
    pico_eval_path: Path,
    retrieval_eval_path: Path,
    answer_eval_path: Path,
    wrapper_outputs_path: Path,
) -> Tuple[bool, Dict[str, object], List[str]]:
    notes: List[str] = []
    if not chapter_md.exists():
        return False, {"chapter_md": str(chapter_md)}, [f"Missing chapter markdown: {chapter_md}"]

    text = chapter_md.read_text(encoding="utf-8")
    required_tokens = [
        "Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest/pico_eval.json",
        "Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest/retrieval_eval.json",
        "Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest/answer_eval.json",
        "Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest/wrapper_outputs.jsonl",
        CHAPTER4_RUN_ID,
    ]
    missing_tokens = [tok for tok in required_tokens if tok not in text]
    for tok in missing_tokens:
        notes.append(f"Chapter markdown missing reference token: {tok}")

    ok = len(missing_tokens) == 0
    details = {
        "chapter_md": str(chapter_md),
        "has_pico_eval_ref": required_tokens[0] not in missing_tokens,
        "has_retrieval_eval_ref": required_tokens[1] not in missing_tokens,
        "has_answer_eval_ref": required_tokens[2] not in missing_tokens,
        "has_wrapper_outputs_ref": required_tokens[3] not in missing_tokens,
        "has_chapter4_run_id_ref": required_tokens[4] not in missing_tokens,
        "pico_eval_path": str(pico_eval_path),
        "retrieval_eval_path": str(retrieval_eval_path),
        "answer_eval_path": str(answer_eval_path),
        "wrapper_outputs_path": str(wrapper_outputs_path),
    }
    return ok, details, notes


def _build_report_md(
    status: str,
    checklist: Dict[str, bool],
    details: Dict[str, Dict[str, object]],
    notes: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# {status}")
    lines.append("")
    lines.append("## Chapter 5 Completion Checklist")
    lines.append(f"- {_bool_icon(checklist['kb_index_built'])} KB index built and manifest valid")
    lines.append(
        f"- {_bool_icon(checklist['wrapper_ran_min_queries'])} Wrapper ran on at least N={details['wrapper']['min_queries_required']} queries"
    )
    lines.append(f"- {_bool_icon(checklist['pico_eval_exists'])} PICO evaluation file exists and is readable")
    lines.append(f"- {_bool_icon(checklist['retrieval_eval_exists'])} Retrieval evaluation file exists and is readable")
    lines.append(f"- {_bool_icon(checklist['answer_eval_exists'])} Answer evaluation file exists and is readable")
    lines.append(
        f"- {_bool_icon(checklist['chapter_text_synced'])} Chapter 5 markdown exists and references Chapter 5 artifacts + frozen Chapter 4 run id"
    )
    lines.append("")
    lines.append("## Key Paths")
    lines.append(f"- KB manifest: `{details['kb'].get('manifest_path')}`")
    lines.append(f"- Wrapper outputs: `{details['wrapper'].get('wrapper_outputs_path')}`")
    lines.append(f"- PICO eval: `{details['pico_eval'].get('path')}`")
    lines.append(f"- Retrieval eval: `{details['retrieval_eval'].get('path')}`")
    lines.append(f"- Answer eval: `{details['answer_eval'].get('path')}`")
    lines.append(f"- Chapter markdown: `{details['chapter_sync'].get('chapter_md')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- wrapper outputs counted: `{details['wrapper'].get('n_wrapper_outputs')}`")
    lines.append(f"- kb chunks: `{details['kb'].get('n_chunks')}`")
    lines.append(f"- kb docs: `{details['kb'].get('n_docs')}`")
    lines.append("")
    if notes:
        lines.append("## Missing / Fixes Required")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
        lines.append("## Minimal Punch List")
        lines.append("1. Build KB index:")
        lines.append(
            "   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/build_kb.py --kb_dir Prototyping_reformat/chapter5_pico_wrapper/data/kb --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest`"
        )
        lines.append("2. Run wrapper on query set:")
        lines.append(
            "   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_wrapper.py --query_file Prototyping_reformat/chapter5_pico_wrapper/data/queries/queries.jsonl --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json --retrieval_k 5 --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest`"
        )
        lines.append("3. Generate evaluations:")
        lines.append(
            "   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_pico.py --pico_gold Prototyping_reformat/chapter5_pico_wrapper/data/queries/pico_gold.jsonl --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest`"
        )
        lines.append(
            "   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_retrieval.py --retrieval_gold Prototyping_reformat/chapter5_pico_wrapper/data/queries/retrieval_gold.jsonl --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json --k_values 1,3,5 --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest`"
        )
        lines.append(
            "   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_answers.py --wrapper_outputs Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest/wrapper_outputs.jsonl --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest`"
        )
        lines.append("4. Update Chapter 5 markdown artifact references and rerun audit.")
    else:
        lines.append("## Notes")
        lines.append("- All required Chapter 5 artifacts are present and readable.")
        lines.append("- Chapter markdown references are synchronized with generated outputs.")
    lines.append("")
    lines.append(f"- frozen Chapter 4 run reference: `{CHAPTER4_RUN_ID}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    workspace_root = _find_workspace_root()
    run_id = args.audit_run_id or f"chapter5_completion_audit_{_now_utc_stamp()}"
    out_dir = args.out_dir.resolve() if args.out_dir else (workspace_root / "results" / run_id).resolve()
    _ensure_dir(out_dir)

    kb_ok, kb_details, kb_notes = _check_kb(args.kb_manifest.resolve())
    wrapper_ok, wrapper_details, wrapper_notes = _check_wrapper_outputs(
        args.wrapper_outputs.resolve(),
        args.run_config.resolve(),
        min_queries=args.min_queries,
    )
    pico_ok, pico_details, pico_notes = _check_eval_file(args.pico_eval.resolve(), "pico_eval")
    retrieval_ok, retrieval_details, retrieval_notes = _check_eval_file(args.retrieval_eval.resolve(), "retrieval_eval")
    answer_ok, answer_details, answer_notes = _check_eval_file(args.answer_eval.resolve(), "answer_eval")
    chapter_ok, chapter_details, chapter_notes = _check_chapter_sync(
        chapter_md=args.chapter_md.resolve(),
        pico_eval_path=args.pico_eval.resolve(),
        retrieval_eval_path=args.retrieval_eval.resolve(),
        answer_eval_path=args.answer_eval.resolve(),
        wrapper_outputs_path=args.wrapper_outputs.resolve(),
    )

    checklist = {
        "kb_index_built": kb_ok,
        "wrapper_ran_min_queries": wrapper_ok,
        "pico_eval_exists": pico_ok,
        "retrieval_eval_exists": retrieval_ok,
        "answer_eval_exists": answer_ok,
        "chapter_text_synced": chapter_ok,
    }
    status = "PASS" if all(checklist.values()) else "FAIL"

    details = {
        "kb": kb_details,
        "wrapper": wrapper_details,
        "pico_eval": pico_details,
        "retrieval_eval": retrieval_details,
        "answer_eval": answer_details,
        "chapter_sync": chapter_details,
    }

    notes = kb_notes + wrapper_notes + pico_notes + retrieval_notes + answer_notes + chapter_notes
    report_text = _build_report_md(status=status, checklist=checklist, details=details, notes=notes)
    report_path = out_dir / "chapter5_completion_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary = {
        "status": status,
        "checklist": checklist,
        "details": details,
        "notes": notes,
        "run_id": run_id,
        "report_path": str(report_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "chapter5_completion_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Chapter 5 completion audit: {status}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
