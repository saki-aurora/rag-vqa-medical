#!/usr/bin/env python3
"""Produce LoRA adapter load-proof artifacts from persisted run folders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from _results_utils import find_limuc_root, write_csv


HASH_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "adapter_config.json",
    "adapter_model.bin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to LIMUC root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="vlm_lora_finetune_mayo",
        help=(
            "Preferred LoRA run folder name/pattern under */results/. "
            "If not found, falls back to any run containing 'lora'."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit adapter folder path. If omitted, script searches "
            "inside the selected run folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output table folder. Default: <LIMUC>/4_reporting/results/tables",
    )
    return parser.parse_args()


def _find_lora_run(dataset_root: Path, run_name: str) -> Path:
    exact = sorted(dataset_root.glob(f"**/results/{run_name}"))
    if exact:
        return exact[0]

    by_pattern = sorted(dataset_root.glob(f"**/results/*{run_name}*"))
    if by_pattern:
        return by_pattern[0]

    fallback = sorted(dataset_root.glob("**/results/*lora*"))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(f"No LoRA-like run folder found under: {dataset_root}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_files(run_dir: Path, adapter_dir: Path | None) -> List[Path]:
    search_roots = [adapter_dir] if adapter_dir else []
    search_roots.extend(
        [
            run_dir,
            run_dir / "lora_adapter",
            run_dir / "checkpoints",
        ]
    )

    found: List[Path] = []
    seen = set()
    for root in search_roots:
        if root is None or not root.exists():
            continue
        root = root.resolve()
        for pattern in HASH_FILE_PATTERNS:
            for path in root.rglob(pattern):
                if path.is_file() and path not in seen:
                    found.append(path)
                    seen.add(path)
    return sorted(found)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else find_limuc_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "4_reporting" / "results" / "tables")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _find_lora_run(dataset_root, args.run_name)
    run_meta_path = run_dir / "run_meta.json"
    run_meta: Dict[str, object] = {}
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    pred_test_path = run_dir / "pred_test.csv"
    test_rows = None
    if pred_test_path.exists():
        with pred_test_path.open("r", encoding="utf-8") as f:
            test_rows = max(0, sum(1 for _ in f) - 1)

    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir else None
    hash_files = _candidate_files(run_dir, adapter_dir)

    hash_rows = []
    for file_path in hash_files:
        hash_rows.append(
            {
                "run_name": run_dir.name,
                "file_path": str(file_path),
                "relative_to_run_dir": str(file_path.relative_to(run_dir.resolve()))
                if run_dir.resolve() in file_path.resolve().parents
                else "",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": _sha256_file(file_path),
            }
        )
    hash_csv = output_dir / "lora_adapter_file_hashes.csv"
    write_csv(
        hash_rows,
        hash_csv,
        field_order=("run_name", "file_path", "relative_to_run_dir", "file_size_bytes", "sha256"),
    )

    adapter_config = None
    for row in hash_rows:
        if row["file_path"].endswith("adapter_config.json"):
            adapter_config = Path(row["file_path"])
            break

    lora_config = {}
    if adapter_config and adapter_config.exists():
        lora_config = json.loads(adapter_config.read_text(encoding="utf-8"))

    lora_param_count = {
        "run_name": run_dir.name,
        "trainable_params_reported": run_meta.get("trainable_params"),
        "all_params_reported": run_meta.get("all_params"),
        "trainable_ratio_reported": run_meta.get("trainable_ratio"),
        "lora_r": lora_config.get("r"),
        "lora_alpha": lora_config.get("lora_alpha"),
        "target_modules": lora_config.get("target_modules"),
        "task_type": lora_config.get("task_type"),
    }
    lora_param_json = output_dir / "lora_param_count.json"
    lora_param_json.write_text(json.dumps(lora_param_count, indent=2), encoding="utf-8")

    has_adapter_files = len(hash_rows) > 0
    has_training_log = (run_dir / "training_history.csv").exists() or (run_dir / "train_log.csv").exists()
    status = "PASS" if (has_adapter_files and has_training_log and test_rows == 1686) else "INCOMPLETE"

    proof_lines = [
        "LORA LOAD PROOF",
        "===============",
        f"Run folder: {run_dir}",
        f"Run name: {run_dir.name}",
        f"run_meta.json present: {'yes' if run_meta_path.exists() else 'no'}",
        f"pred_test.csv rows: {test_rows if test_rows is not None else 'missing'}",
        f"adapter-related files found: {len(hash_rows)}",
        f"training log found: {'yes' if has_training_log else 'no'}",
        f"adapter_config found: {'yes' if adapter_config is not None else 'no'}",
        f"Status: {status}",
        "",
        "Run meta excerpt:",
        json.dumps(run_meta, indent=2),
        "",
        "LoRA config excerpt:",
        json.dumps(lora_config, indent=2) if lora_config else "{}",
        "",
        f"Adapter file hash table: {hash_csv}",
        f"LoRA param-count file: {lora_param_json}",
    ]
    proof_path = output_dir / "lora_load_proof.txt"
    proof_path.write_text("\n".join(proof_lines), encoding="utf-8")

    print(f"Selected LoRA run: {run_dir}")
    print(f"Status: {status}")
    print(f"Adapter files hashed: {len(hash_rows)}")
    print(f"Wrote: {hash_csv}")
    print(f"Wrote: {lora_param_json}")
    print(f"Wrote: {proof_path}")


if __name__ == "__main__":
    main()

