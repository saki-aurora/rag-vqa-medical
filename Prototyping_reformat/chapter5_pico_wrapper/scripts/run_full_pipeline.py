#!/usr/bin/env python3
"""Run one-command Chapter 5 pipeline (queryset -> KB -> wrapper -> evals -> audit)."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_id_with_tag(tag: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag.strip())
    clean = clean.strip("_-") or "latest"
    return clean


@dataclass
class StepResult:
    name: str
    command: List[str]
    log_path: Path
    returncode: int
    started_utc: str
    ended_utc: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "command": self.command,
            "command_pretty": " ".join(shlex.quote(x) for x in self.command),
            "log_path": str(self.log_path),
            "returncode": self.returncode,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
        }


def _run_step(*, name: str, command: List[str], log_path: Path, dry_run: bool) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    if dry_run:
        log_path.write_text(
            "[dry_run] command not executed\n" + " ".join(shlex.quote(x) for x in command) + "\n",
            encoding="utf-8",
        )
        ended = datetime.now(timezone.utc).isoformat()
        return StepResult(
            name=name,
            command=command,
            log_path=log_path,
            returncode=0,
            started_utc=started,
            ended_utc=ended,
        )

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"$ {' '.join(shlex.quote(x) for x in command)}\n\n")
        lf.flush()
        completed = subprocess.run(command, stdout=lf, stderr=subprocess.STDOUT, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step '{name}' failed (returncode={completed.returncode}). "
            f"See log: {log_path}"
        )

    return StepResult(
        name=name,
        command=command,
        log_path=log_path,
        returncode=completed.returncode,
        started_utc=started,
        ended_utc=ended,
    )


def parse_args() -> argparse.Namespace:
    workspace_root = _find_workspace_root()
    repo_root = _find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for all steps.")
    parser.add_argument("--tag", type=str, default="pass4_latest", help="Artifact tag suffix.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--kb_dir", type=Path, default=workspace_root / "data" / "kb")
    parser.add_argument("--query_dir", type=Path, default=workspace_root / "data" / "queries")
    parser.add_argument("--results_root", type=Path, default=workspace_root / "results")
    parser.add_argument(
        "--chapter_md",
        type=Path,
        default=repo_root / "Thesis" / "markdown" / "05_chapter_5_genai_wrapper_pico.md",
    )

    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--n_pico_gold", type=int, default=20)
    parser.add_argument("--skip_queryset", action="store_true")

    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "llm"])
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument("--retrieval_backend", type=str, default="hybrid", choices=["keyword", "tfidf", "hybrid"])
    parser.add_argument("--disable_rerank", action="store_true")
    parser.add_argument("--rerank_pool", type=int, default=20)
    parser.add_argument("--rerank_alpha", type=float, default=0.20)
    parser.add_argument("--min_top_score_for_answer", type=float, default=0.18)
    parser.add_argument("--min_mean_score_for_answer", type=float, default=0.12)
    parser.add_argument("--min_retrieved_for_answer", type=int, default=2)

    parser.add_argument("--k_values", type=str, default="1,3,5")
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--min_overlap_ratio_strict", type=float, default=0.25)
    parser.add_argument("--min_overlap_terms_strict", type=int, default=3)
    parser.add_argument("--min_queries_for_audit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = _find_workspace_root()

    tag = _run_id_with_tag(args.tag)
    results_root = args.results_root.resolve()
    kb_out = results_root / f"kb_build_{tag}"
    wrapper_out = results_root / f"wrapper_eval_{tag}"
    eval_out = results_root / f"eval_{tag}"
    audit_out = results_root / f"chapter5_completion_audit_{tag}"
    pipeline_out = results_root / f"pipeline_{tag}"
    logs_dir = pipeline_out / "logs"
    pipeline_out.mkdir(parents=True, exist_ok=True)

    query_file = args.query_dir.resolve() / "queries.jsonl"
    pico_gold = args.query_dir.resolve() / "pico_gold.jsonl"
    retrieval_gold = args.query_dir.resolve() / "retrieval_gold.jsonl"
    kb_manifest = kb_out / "kb_manifest.json"
    wrapper_outputs = wrapper_out / "wrapper_outputs.jsonl"
    run_config = wrapper_out / "run_config.json"
    pico_eval = eval_out / "pico_eval.json"
    retrieval_eval = eval_out / "retrieval_eval.json"
    answer_eval = eval_out / "answer_eval.json"

    steps: List[StepResult] = []
    py = str(Path(args.python).resolve())

    if not args.skip_queryset:
        steps.append(
            _run_step(
                name="make_queryset",
                command=[
                    py,
                    str(workspace_root / "scripts" / "make_queryset.py"),
                    "--out_dir",
                    str(args.query_dir.resolve()),
                    "--n_queries",
                    str(args.n_queries),
                    "--n_pico_gold",
                    str(args.n_pico_gold),
                ],
                log_path=logs_dir / "01_make_queryset.log",
                dry_run=args.dry_run,
            )
        )

    steps.append(
        _run_step(
            name="build_kb",
            command=[
                py,
                str(workspace_root / "scripts" / "build_kb.py"),
                "--kb_dir",
                str(args.kb_dir.resolve()),
                "--out_dir",
                str(kb_out),
                "--seed",
                str(args.seed),
            ],
            log_path=logs_dir / "02_build_kb.log",
            dry_run=args.dry_run,
        )
    )

    run_wrapper_cmd = [
        py,
        str(workspace_root / "scripts" / "run_wrapper.py"),
        "--query_file",
        str(query_file),
        "--manifest_path",
        str(kb_manifest),
        "--mode",
        args.mode,
        "--retrieval_k",
        str(args.retrieval_k),
        "--retrieval_backend",
        args.retrieval_backend,
        "--rerank_pool",
        str(args.rerank_pool),
        "--rerank_alpha",
        str(args.rerank_alpha),
        "--min_top_score_for_answer",
        str(args.min_top_score_for_answer),
        "--min_mean_score_for_answer",
        str(args.min_mean_score_for_answer),
        "--min_retrieved_for_answer",
        str(args.min_retrieved_for_answer),
        "--out_dir",
        str(wrapper_out),
    ]
    if args.disable_rerank:
        run_wrapper_cmd.append("--disable_rerank")
    steps.append(
        _run_step(
            name="run_wrapper",
            command=run_wrapper_cmd,
            log_path=logs_dir / "03_run_wrapper.log",
            dry_run=args.dry_run,
        )
    )

    steps.append(
        _run_step(
            name="eval_pico",
            command=[
                py,
                str(workspace_root / "scripts" / "eval_pico.py"),
                "--pico_gold",
                str(pico_gold),
                "--out_dir",
                str(eval_out),
            ],
            log_path=logs_dir / "04_eval_pico.log",
            dry_run=args.dry_run,
        )
    )

    eval_retrieval_cmd = [
        py,
        str(workspace_root / "scripts" / "eval_retrieval.py"),
        "--retrieval_gold",
        str(retrieval_gold),
        "--manifest_path",
        str(kb_manifest),
        "--k_values",
        args.k_values,
        "--retrieval_backend",
        args.retrieval_backend,
        "--rerank_pool",
        str(args.rerank_pool),
        "--rerank_alpha",
        str(args.rerank_alpha),
        "--bootstrap_iters",
        str(args.bootstrap_iters),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(eval_out),
    ]
    if args.disable_rerank:
        eval_retrieval_cmd.append("--disable_rerank")
    steps.append(
        _run_step(
            name="eval_retrieval",
            command=eval_retrieval_cmd,
            log_path=logs_dir / "05_eval_retrieval.log",
            dry_run=args.dry_run,
        )
    )

    steps.append(
        _run_step(
            name="eval_answers",
            command=[
                py,
                str(workspace_root / "scripts" / "eval_answers.py"),
                "--wrapper_outputs",
                str(wrapper_outputs),
                "--min_overlap_ratio_strict",
                str(args.min_overlap_ratio_strict),
                "--min_overlap_terms_strict",
                str(args.min_overlap_terms_strict),
                "--out_dir",
                str(eval_out),
            ],
            log_path=logs_dir / "06_eval_answers.log",
            dry_run=args.dry_run,
        )
    )

    steps.append(
        _run_step(
            name="completion_audit",
            command=[
                py,
                str(workspace_root / "scripts" / "chapter5_completion_audit.py"),
                "--kb_manifest",
                str(kb_manifest),
                "--wrapper_outputs",
                str(wrapper_outputs),
                "--run_config",
                str(run_config),
                "--pico_eval",
                str(pico_eval),
                "--retrieval_eval",
                str(retrieval_eval),
                "--answer_eval",
                str(answer_eval),
                "--chapter_md",
                str(args.chapter_md.resolve()),
                "--min_queries",
                str(args.min_queries_for_audit),
                "--out_dir",
                str(audit_out),
                "--audit_run_id",
                f"chapter5_completion_audit_{tag}_{_utc_now_compact()}",
            ],
            log_path=logs_dir / "07_completion_audit.log",
            dry_run=args.dry_run,
        )
    )

    # best-effort environment freeze for the overall pipeline
    if not args.dry_run:
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from pico_wrapper.utils_io import freeze_environment

        freeze_environment(pipeline_out / "environment.txt")

    summary = {
        "task": "chapter5_full_pipeline",
        "tag": tag,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "python": py,
        "workspace_root": str(workspace_root),
        "paths": {
            "query_dir": str(args.query_dir.resolve()),
            "results_root": str(results_root),
            "kb_out": str(kb_out),
            "wrapper_out": str(wrapper_out),
            "eval_out": str(eval_out),
            "audit_out": str(audit_out),
            "pipeline_out": str(pipeline_out),
        },
        "artifacts": {
            "query_file": str(query_file),
            "pico_gold": str(pico_gold),
            "retrieval_gold": str(retrieval_gold),
            "kb_manifest": str(kb_manifest),
            "wrapper_outputs": str(wrapper_outputs),
            "run_config": str(run_config),
            "pico_eval": str(pico_eval),
            "retrieval_eval": str(retrieval_eval),
            "answer_eval": str(answer_eval),
        },
        "steps": [s.to_dict() for s in steps],
    }

    summary_path = pipeline_out / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Pipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
