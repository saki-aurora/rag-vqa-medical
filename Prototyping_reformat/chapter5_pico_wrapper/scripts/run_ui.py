#!/usr/bin/env python3
"""Run a local browser UI for the Chapter 5 PICO wrapper (no extra dependencies)."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_html(default_manifest_path: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chapter 5 PICO Wrapper UI</title>
  <style>
    :root {{
      --bg:#f5f7fb;
      --card:#ffffff;
      --text:#13213a;
      --muted:#51607c;
      --accent:#0f766e;
      --line:#d9e0ee;
      --danger:#b91c1c;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #d8efe7 0%, var(--bg) 45%);
    }}
    .wrap {{
      max-width: 980px;
      margin: 20px auto;
      padding: 0 14px 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 1.35rem;
    }}
    .hint {{
      color: var(--muted);
      margin: 0 0 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 26px rgba(16, 32, 66, 0.08);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
    label {{
      display: block;
      font-size: 0.92rem;
      margin-bottom: 5px;
      color: var(--muted);
      font-weight: 600;
    }}
    input, select, textarea {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #c9d4ea;
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 0.95rem;
      background: #fbfdff;
    }}
    textarea {{ min-height: 118px; resize: vertical; }}
    .row {{ margin-bottom: 10px; }}
    .btn {{
      border: 0;
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(120deg, #0f766e, #0b5f9e);
    }}
    .status {{
      margin-top: 10px;
      font-size: 0.92rem;
    }}
    .status.error {{ color: var(--danger); }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.86rem;
      line-height: 1.45;
      background: #0b1220;
      color: #dce7ff;
      padding: 11px;
      border-radius: 10px;
      overflow: auto;
      max-height: 440px;
    }}
    .result {{
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }}
    .foot {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Chapter 5 PICO Wrapper UI</h1>
    <p class="hint">Local UI for the wrapper pipeline: query -> PICO -> retrieval -> structured answer.</p>
    <div class="card">
      <div class="row">
        <label for="query">Physician Query</label>
        <textarea id="query">For adults with ulcerative colitis, does biologic therapy versus standard care improve clinical remission at 12 weeks?</textarea>
      </div>
      <div class="grid">
        <div class="row">
          <label for="manifest_path">KB Manifest Path</label>
          <input id="manifest_path" value="{default_manifest_path}" />
        </div>
        <div class="row">
          <label for="mode">Mode</label>
          <select id="mode">
            <option value="baseline">baseline</option>
            <option value="llm">llm (fallback if unavailable)</option>
          </select>
        </div>
        <div class="row">
          <label for="retrieval_k">Retrieval K</label>
          <input id="retrieval_k" type="number" min="1" value="5" />
        </div>
        <div class="row">
          <label for="severity_json">Severity JSON (optional)</label>
          <input id="severity_json" placeholder='{{"mes_pred":2,"confidence":0.78,"run_id":"chapter4_run"}}' />
        </div>
      </div>
      <button class="btn" id="run_btn">Run Wrapper</button>
      <div class="status" id="status"></div>
      <div class="result">
        <div>
          <label>Run Info</label>
          <pre id="run_info"></pre>
        </div>
        <div>
          <label>Wrapper Output</label>
          <pre id="output"></pre>
        </div>
      </div>
      <div class="foot">Safety note: this tool is decision-support only and does not provide patient-specific dosing.</div>
    </div>
  </div>
<script>
async function runWrapper() {{
  const status = document.getElementById("status");
  const runInfo = document.getElementById("run_info");
  const output = document.getElementById("output");
  status.className = "status";
  status.textContent = "Running...";
  runInfo.textContent = "";
  output.textContent = "";

  const payload = {{
    query: document.getElementById("query").value,
    manifest_path: document.getElementById("manifest_path").value,
    mode: document.getElementById("mode").value,
    retrieval_k: Number(document.getElementById("retrieval_k").value || 5),
    severity_json: document.getElementById("severity_json").value
  }};

  try {{
    const res = await fetch("/api/run", {{
      method: "POST",
      headers: {{"Content-Type":"application/json"}},
      body: JSON.stringify(payload)
    }});
    const body = await res.json();
    if (!res.ok) {{
      status.className = "status error";
      status.textContent = body.error || "Request failed";
      return;
    }}
    status.textContent = "Done";
    runInfo.textContent = JSON.stringify(body.info, null, 2);
    output.textContent = JSON.stringify(body.output, null, 2);
  }} catch (err) {{
    status.className = "status error";
    status.textContent = "Error: " + err;
  }}
}}
document.getElementById("run_btn").addEventListener("click", runWrapper);
</script>
</body>
</html>
"""


class UiServer:
    def __init__(self, manifest_path: Path, out_dir: Path, default_mode: str, default_k: int) -> None:
        self.manifest_path = manifest_path
        self.out_dir = out_dir
        self.default_mode = default_mode
        self.default_k = default_k

    def make_handler(self):
        manifest_path = self.manifest_path
        out_dir = self.out_dir
        default_mode = self.default_mode
        default_k = self.default_k

        workspace_root = _find_workspace_root()
        import sys

        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from pico_wrapper.utils_io import generate_run_id
        from pico_wrapper.wrapper import run_wrapper

        class Handler(BaseHTTPRequestHandler):
            def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _html_response(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/" or self.path.startswith("/?"):
                    self._html_response(_build_html(str(manifest_path)))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/run":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return
                try:
                    cl = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(cl).decode("utf-8")
                    req = json.loads(raw) if raw else {}
                    query = str(req.get("query", "")).strip()
                    if not query:
                        self._json_response(HTTPStatus.BAD_REQUEST, {"error": "query is required"})
                        return

                    manifest_raw = str(req.get("manifest_path", "")).strip()
                    req_manifest = Path(manifest_raw).resolve() if manifest_raw else manifest_path
                    if not req_manifest.exists():
                        self._json_response(
                            HTTPStatus.BAD_REQUEST,
                            {"error": f"kb manifest not found: {req_manifest}"},
                        )
                        return

                    mode = str(req.get("mode", default_mode)).strip().lower() or "baseline"
                    if mode not in {"baseline", "llm"}:
                        mode = "baseline"

                    retrieval_k = int(req.get("retrieval_k", default_k))
                    if retrieval_k < 1:
                        retrieval_k = default_k

                    severity: Optional[Dict[str, Any]] = None
                    severity_raw = str(req.get("severity_json", "")).strip()
                    if severity_raw:
                        severity = json.loads(severity_raw)

                    run_id = generate_run_id("chapter5_ui")
                    output, info = run_wrapper(
                        query=query,
                        manifest_path=req_manifest,
                        run_id=run_id,
                        retrieval_k=retrieval_k,
                        mode=mode,
                        severity=severity,
                    )

                    _append_jsonl(out_dir / "wrapper_outputs.jsonl", output.to_dict())
                    _append_jsonl(out_dir / "wrapper_run_infos.jsonl", info.to_dict())
                    _append_jsonl(
                        out_dir / "ui_requests.jsonl",
                        {
                            "run_id": run_id,
                            "query": query,
                            "manifest_path": str(req_manifest),
                            "mode": mode,
                            "retrieval_k": retrieval_k,
                            "has_severity_json": severity is not None,
                        },
                    )
                    self._json_response(
                        HTTPStatus.OK,
                        {
                            "run_id": run_id,
                            "info": info.to_dict(),
                            "output": output.to_dict(),
                        },
                    )
                except json.JSONDecodeError as exc:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
                except Exception as exc:  # noqa: BLE001
                    self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                # Keep terminal output concise.
                return

        return Handler


def parse_args() -> argparse.Namespace:
    root = _find_workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=root / "results" / "kb_build_latest" / "kb_manifest.json",
        help="Default KB manifest path shown in UI.",
    )
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "llm"])
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "ui_latest",
        help="Directory where UI requests and outputs are appended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"kb manifest not found: {manifest_path}\n"
            "Run build_kb.py first or pass --manifest_path."
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ui = UiServer(
        manifest_path=manifest_path,
        out_dir=out_dir,
        default_mode=args.mode,
        default_k=args.retrieval_k,
    )
    handler = ui.make_handler()
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Chapter 5 UI running at http://{args.host}:{args.port}")
    print(f"Default manifest: {manifest_path}")
    print(f"Output log dir: {out_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
