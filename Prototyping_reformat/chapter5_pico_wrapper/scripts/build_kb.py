#!/usr/bin/env python3
"""Build Chapter 5 KB chunks and retrieval index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def parse_args() -> argparse.Namespace:
    root = _find_workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kb_dir",
        type=Path,
        default=root / "data" / "kb",
        help="Directory containing KB source docs (.md/.txt).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "kb_build_latest",
        help="Output directory to write chunks/index/manifest.",
    )
    parser.add_argument("--max_words", type=int, default=180)
    parser.add_argument("--overlap_words", type=int, default=30)
    parser.add_argument("--min_words", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    workspace_root = _find_workspace_root()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    from pico_wrapper.kb_ingest import build_kb_index
    from pico_wrapper.utils_io import freeze_environment

    summary = build_kb_index(
        kb_dir=args.kb_dir,
        out_dir=args.out_dir,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
        random_seed=args.seed,
    )
    freeze_environment(Path(args.out_dir) / "environment.txt")

    print("KB build complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
