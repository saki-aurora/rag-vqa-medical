"""I/O and run-level reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
import string
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_run_id(prefix: str = "chapter5_wrapper") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{prefix}_{ts}_{suffix}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_environment(path: Path) -> None:
    """Persist environment details for reproducibility.

    Best effort:
    1) python executable
    2) uname
    3) pip freeze
    """
    ensure_dir(path.parent)
    lines = []
    lines.append(f"timestamp_utc={datetime.now(timezone.utc).isoformat()}")
    lines.append(f"python={os.sys.executable}")

    try:
        uname = subprocess.check_output(["uname", "-a"], text=True).strip()
        lines.append(f"uname={uname}")
    except Exception:
        lines.append("uname=<unavailable>")

    try:
        pip_freeze = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
        lines.append("")
        lines.append("[pip_freeze]")
        lines.append(pip_freeze.strip())
    except Exception:
        lines.append("")
        lines.append("[pip_freeze]")
        lines.append("<unavailable>")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

