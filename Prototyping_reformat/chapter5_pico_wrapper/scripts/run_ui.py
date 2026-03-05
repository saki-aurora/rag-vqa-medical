#!/usr/bin/env python3
"""Run a local browser UI for the Chapter 5 PICO wrapper (no extra dependencies)."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shlex
import subprocess
import traceback
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import parse_qs, urlparse


MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl_tail(path: Path, limit: int = 20) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: deque[Dict[str, Any]] = deque(maxlen=max(1, int(limit)))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "upload.bin").strip())
    out = out.strip("._")
    return out or "upload.bin"


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _decode_image_upload(payload: Mapping[str, Any]) -> tuple[bytes, str, str]:
    raw_name = str(payload.get("name", "upload.bin")).strip()
    mime = str(payload.get("mime_type", "application/octet-stream")).strip()
    b64 = str(payload.get("data_base64", "")).strip()
    if not b64:
        raise ValueError("image_upload.data_base64 is required")

    if "base64," in b64:
        b64 = b64.split("base64,", 1)[1].strip()

    try:
        blob = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid base64 image payload: {exc}") from exc

    if len(blob) == 0:
        raise ValueError("uploaded image is empty")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ValueError(f"uploaded image exceeds limit ({MAX_UPLOAD_BYTES} bytes)")

    return blob, _sanitize_filename(raw_name), mime


def _save_upload(*, out_dir: Path, run_id: str, filename: str, blob: bytes) -> Path:
    uploads_dir = out_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = uploads_dir / f"{ts}_{run_id}_{filename}"
    path.write_bytes(blob)
    return path


def _extract_probs(row: Mapping[str, str]) -> Optional[List[float]]:
    for prefix in ["prob_", "prob_mayo_", "p"]:
        vals: List[float] = []
        ok = True
        for i in range(4):
            key = f"{prefix}{i}"
            if key not in row or str(row.get(key, "")).strip() == "":
                ok = False
                break
            vals.append(_as_float(row.get(key), default=0.0))
        if ok:
            return vals
    return None


def _load_lookup_predictions(csv_path: Optional[Path], run_id: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if csv_path is None:
        return out
    path = csv_path.resolve()
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_raw = row.get("y_pred") or row.get("pred_label")
            if pred_raw is None or str(pred_raw).strip() == "":
                continue
            pred = _as_int(pred_raw, default=-1)
            if pred < 0 or pred > 3:
                continue
            probs = _extract_probs(row)
            conf = max(probs) if probs else None
            img_id = str(row.get("img_id", "")).strip()
            img_path = str(row.get("image_path", "")).strip()
            cues = [f"lookup_source:{path.name}"]
            if img_id:
                cues.append(f"img_id:{img_id}")
            sev = {
                "mes_pred": pred,
                "mes_probs": probs,
                "confidence": conf,
                "quality_flag": "lookup",
                "cues": cues,
                "model_version": "chapter4_lookup",
                "run_id": run_id,
                "timestamp": _utc_now_iso(),
            }

            candidates = set()
            if img_path:
                candidates.add(img_path.lower())
                candidates.add(Path(img_path).name.lower())
            if img_id:
                candidates.add(img_id.lower())
            for key in candidates:
                if key and key not in out:
                    out[key] = sev
    return out


def _find_lookup_severity(
    lookup: Mapping[str, Dict[str, Any]],
    *,
    image_path: Optional[Path],
    original_filename: Optional[str],
) -> Optional[Dict[str, Any]]:
    keys: List[str] = []
    if image_path is not None:
        keys.append(str(image_path).lower())
        keys.append(image_path.name.lower())
        keys.append(image_path.stem.lower())
    if original_filename:
        p = Path(original_filename)
        keys.append(original_filename.lower())
        keys.append(p.name.lower())
        keys.append(p.stem.lower())

    for key in keys:
        if key in lookup:
            return dict(lookup[key])
    return None


def _predict_severity_with_command(
    *,
    cmd_template: str,
    image_path: Path,
    workspace_root: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    if "{image_path}" not in cmd_template:
        raise ValueError("severity_predict_cmd must include '{image_path}' placeholder")

    cmd = cmd_template.format(
        image_path=str(image_path.resolve()),
        workspace_root=str(workspace_root.resolve()),
    )
    parts = shlex.split(cmd)
    if not parts:
        raise ValueError("severity_predict_cmd is empty after parsing")

    completed = subprocess.run(
        parts,
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout_sec)),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "severity predictor command failed"
            f" (code={completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise RuntimeError("severity predictor command returned empty stdout")

    parsed: Optional[Dict[str, Any]] = None
    candidates = [stdout]
    candidates.extend([ln.strip() for ln in stdout.splitlines() if ln.strip()])
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed = obj
            break
    if parsed is None:
        raise RuntimeError("severity predictor command did not return valid JSON object")

    return parsed


def _default_lookup_csv() -> Optional[Path]:
    repo_root = _find_workspace_root().parents[1]
    candidate = repo_root / "Prototyping_reformat" / "DatasetAnalysis" / "LIMUC" / "2_supervised_finetuning" / "results" / "finetune_resnet50" / "pred_test.csv"
    if candidate.exists():
        return candidate
    return None


def _build_html(default_manifest_path: str) -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Chapter 5 Clinical Wrapper Console</title>
  <style>
    :root {{
      --ink:#141b30;
      --muted:#53607c;
      --paper:#f5f7fb;
      --card:#ffffff;
      --line:#d5deef;
      --aqua:#0b7b74;
      --blue:#1e4fa6;
      --warn:#9a5d00;
      --warn-bg:#fff4dd;
      --danger:#7f1414;
      --danger-bg:#ffe8e8;
      --ok:#1f6a3d;
      --ok-bg:#e8f8ef;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Space Grotesk","Manrope","Segoe UI",sans-serif;
      background:
        radial-gradient(900px 420px at 88% -10%, #d6e9ff 0%, transparent 60%),
        radial-gradient(850px 430px at -8% 102%, #d4efe7 0%, transparent 60%),
        var(--paper);
    }}
    .shell {{
      max-width:1280px;
      margin:16px auto 28px;
      padding:0 12px;
      display:grid;
      grid-template-columns:minmax(0, 2fr) minmax(340px, 1fr);
      gap:12px;
    }}
    @media (max-width: 1080px) {{
      .shell {{ grid-template-columns:1fr; }}
    }}
    .card {{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:14px;
      padding:14px;
      box-shadow:0 14px 32px rgba(11,22,52,0.08);
    }}
    .title {{
      margin:0;
      font-size:1.2rem;
      letter-spacing:0.02em;
    }}
    .subtitle {{
      margin:6px 0 0;
      color:var(--muted);
      font-size:0.92rem;
    }}
    .row {{ margin-top:10px; }}
    label {{
      display:block;
      margin:0 0 5px;
      color:var(--muted);
      font-size:0.86rem;
      font-weight:700;
      text-transform:uppercase;
      letter-spacing:0.04em;
    }}
    input, select, textarea {{
      width:100%;
      border:1px solid #c2d0ea;
      border-radius:10px;
      background:#fbfdff;
      padding:9px 10px;
      font-size:0.93rem;
      color:var(--ink);
      font-family:"Manrope","Segoe UI",sans-serif;
    }}
    textarea {{ min-height:110px; resize:vertical; }}
    .grid2 {{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:8px;
    }}
    .grid3 {{
      display:grid;
      grid-template-columns:1fr 1fr 1fr;
      gap:8px;
    }}
    @media (max-width: 760px) {{
      .grid2, .grid3 {{ grid-template-columns:1fr; }}
    }}
    .btnbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
    .btn {{
      border:0;
      border-radius:10px;
      padding:9px 14px;
      font-size:0.9rem;
      font-weight:700;
      color:#fff;
      cursor:pointer;
      background:linear-gradient(130deg, var(--aqua), var(--blue));
    }}
    .btn.secondary {{ background:linear-gradient(130deg, #394259, #2a3045); }}
    .btn.ghost {{
      color:var(--ink);
      background:#e8edf8;
    }}
    .status {{ margin-top:8px; font-size:0.9rem; color:var(--muted); }}
    .status.error {{ color:var(--danger); font-weight:700; }}
    .status.ok {{ color:var(--ok); font-weight:700; }}
    .banner {{
      margin-top:10px;
      border-radius:10px;
      padding:10px 11px;
      border:1px solid transparent;
      display:none;
      font-size:0.9rem;
      line-height:1.45;
    }}
    .banner.warning {{ background:var(--warn-bg); border-color:#f4cf90; color:var(--warn); display:block; }}
    .banner.danger {{ background:var(--danger-bg); border-color:#f2b0b0; color:var(--danger); display:block; }}
    .banner.info {{ background:var(--ok-bg); border-color:#b7e5cc; color:var(--ok); display:block; }}
    .result-grid {{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:9px;
      margin-top:10px;
    }}
    @media (max-width: 860px) {{
      .result-grid {{ grid-template-columns:1fr; }}
    }}
    pre {{
      margin:0;
      padding:10px;
      border-radius:10px;
      border:1px solid #223151;
      background:#0f1628;
      color:#dce8ff;
      font-size:0.82rem;
      line-height:1.45;
      white-space:pre-wrap;
      word-break:break-word;
      max-height:320px;
      overflow:auto;
      font-family:"IBM Plex Mono","JetBrains Mono",monospace;
    }}
    .list {{
      margin:0;
      padding-left:18px;
      font-size:0.88rem;
      line-height:1.45;
    }}
    .list li {{ margin-bottom:6px; }}
    .history-item {{
      border:1px solid var(--line);
      border-radius:10px;
      padding:9px;
      margin-top:8px;
      background:#fbfdff;
      cursor:pointer;
    }}
    .history-item:hover {{ border-color:#8ea7d9; }}
    .history-meta {{ color:var(--muted); font-size:0.78rem; margin-bottom:4px; }}
    .mono {{ font-family:"IBM Plex Mono","JetBrains Mono",monospace; }}
    .tiny {{ font-size:0.78rem; color:var(--muted); }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <section class=\"card\">
      <h1 class=\"title\">Chapter 5 Clinical Wrapper Console</h1>
      <p class=\"subtitle\">Query -> PICO -> hybrid retrieval -> citation-grounded synthesis with safety controls.</p>

      <div class=\"row\">
        <label for=\"query\">Physician Query</label>
        <textarea id=\"query\">For adults with ulcerative colitis, does biologic therapy versus standard care improve clinical remission at 12 weeks?</textarea>
      </div>

      <div class=\"grid2\">
        <div class=\"row\">
          <label for=\"manifest_path\">KB Manifest Path</label>
          <input id=\"manifest_path\" value=\"__DEFAULT_MANIFEST_PATH__\" />
        </div>
        <div class=\"row\">
          <label for=\"mode\">Mode</label>
          <select id=\"mode\">
            <option value=\"baseline\">baseline</option>
            <option value=\"llm\">llm (fallback if unavailable)</option>
          </select>
        </div>
      </div>

      <div class=\"grid3\">
        <div class=\"row\">
          <label for=\"retrieval_k\">Retrieval K</label>
          <input id=\"retrieval_k\" type=\"number\" min=\"1\" value=\"5\" />
        </div>
        <div class=\"row\">
          <label for=\"retrieval_backend\">Retrieval Backend</label>
          <select id=\"retrieval_backend\">
            <option value=\"\">manifest default</option>
            <option value=\"hybrid\" selected>hybrid</option>
            <option value=\"tfidf\">tfidf</option>
            <option value=\"keyword\">keyword</option>
          </select>
        </div>
        <div class=\"row\">
          <label for=\"rerank_alpha\">Rerank Alpha</label>
          <input id=\"rerank_alpha\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.20\" />
        </div>
      </div>

      <div class=\"grid3\">
        <div class=\"row\">
          <label for=\"rerank_pool\">Rerank Pool</label>
          <input id=\"rerank_pool\" type=\"number\" min=\"1\" value=\"20\" />
        </div>
        <div class=\"row\">
          <label for=\"min_top\">Min Top Score</label>
          <input id=\"min_top\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.18\" />
        </div>
        <div class=\"row\">
          <label for=\"min_mean\">Min Mean Score</label>
          <input id=\"min_mean\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.12\" />
        </div>
      </div>

      <div class=\"grid3\">
        <div class=\"row\">
          <label for=\"min_retrieved\">Min Retrieved</label>
          <input id=\"min_retrieved\" type=\"number\" min=\"1\" value=\"2\" />
        </div>
        <div class=\"row\">
          <label for=\"disable_rerank\">Rerank</label>
          <select id=\"disable_rerank\">
            <option value=\"false\" selected>enabled</option>
            <option value=\"true\">disabled</option>
          </select>
        </div>
        <div class=\"row\">
          <label for=\"severity_predictor\">Severity Predictor</label>
          <select id=\"severity_predictor\">
            <option value=\"lookup\" selected>lookup (Chapter 4 predictions)</option>
            <option value=\"command\">command hook</option>
          </select>
        </div>
      </div>

      <div class=\"grid2\">
        <div class=\"row\">
          <label for=\"image_file\">Image Upload (optional)</label>
          <input id=\"image_file\" type=\"file\" accept=\"image/*\" />
        </div>
        <div class=\"row\">
          <label for=\"severity_json\">Severity JSON (optional)</label>
          <input id=\"severity_json\" placeholder='{"mes_pred":2,"confidence":0.78,"run_id":"chapter4_run"}' />
        </div>
      </div>

      <div class=\"btnbar\">
        <button class=\"btn secondary\" id=\"predict_btn\">Predict Severity</button>
        <button class=\"btn\" id=\"run_btn\">Run Wrapper</button>
        <button class=\"btn ghost\" id=\"export_json_btn\">Export JSON</button>
        <button class=\"btn ghost\" id=\"export_md_btn\">Export Markdown</button>
      </div>

      <div class=\"status\" id=\"status\"></div>
      <div class=\"banner\" id=\"safety_banner\"></div>
      <div class=\"row tiny mono\" id=\"predict_meta\"></div>

      <div class=\"result-grid\">
        <div>
          <label>PICO Parse</label>
          <pre id=\"pico_view\"></pre>
        </div>
        <div>
          <label>Run Info</label>
          <pre id=\"run_info\"></pre>
        </div>
        <div>
          <label>Claims + Citations</label>
          <pre id=\"claims_view\"></pre>
        </div>
        <div>
          <label>Evidence</label>
          <pre id=\"evidence_view\"></pre>
        </div>
        <div>
          <label>Wrapper Output (Full JSON)</label>
          <pre id=\"output_json\"></pre>
        </div>
        <div>
          <label>Markdown Report</label>
          <pre id=\"report_md\"></pre>
        </div>
      </div>
      <div class=\"tiny\" style=\"margin-top:10px;\">
        Safety note: decision-support only. No patient-specific dosing recommendations are provided.
      </div>
    </section>

    <aside class=\"card\">
      <h2 class=\"title\">Session History</h2>
      <p class=\"subtitle\">Recent UI runs with quick recall.</p>
      <div class=\"btnbar\">
        <button class=\"btn ghost\" id=\"refresh_history_btn\">Refresh</button>
      </div>
      <div id=\"history_list\"></div>
    </aside>
  </div>
<script>
let lastResult = null;
let historyRows = [];

function setStatus(msg, kind="") {
  const status = document.getElementById("status");
  status.className = "status" + (kind ? " " + kind : "");
  status.textContent = msg || "";
}

function setSafetyBanner(alertObj) {
  const el = document.getElementById("safety_banner");
  el.className = "banner";
  el.textContent = "";
  if (!alertObj) {
    return;
  }
  const level = alertObj.level || "info";
  const title = alertObj.title || "Safety";
  const msg = alertObj.message || "";
  el.className = "banner " + level;
  el.textContent = title + ": " + msg;
}

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function fileToPayload(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const raw = String(reader.result || "");
      const base64 = raw.includes("base64,") ? raw.split("base64,")[1] : raw;
      resolve({
        name: file.name,
        mime_type: file.type || "application/octet-stream",
        data_base64: base64
      });
    };
    reader.onerror = () => reject(reader.error || new Error("failed to read file"));
    reader.readAsDataURL(file);
  });
}

function renderRun(result) {
  const info = (result && result.info) || {};
  const output = (result && result.output) || {};

  document.getElementById("run_info").textContent = JSON.stringify(info, null, 2);
  document.getElementById("output_json").textContent = JSON.stringify(output, null, 2);
  document.getElementById("pico_view").textContent = JSON.stringify(output.pico || {}, null, 2);

  const claims = Array.isArray(output.claims) ? output.claims : [];
  const claimsLines = claims.map((c, i) => {
    const ids = Array.isArray(c.citation_ids) ? c.citation_ids.join(", ") : "";
    return `${i + 1}. ${c.text || ""}${ids ? ` [${ids}]` : ""}`;
  });
  document.getElementById("claims_view").textContent = claimsLines.length ? claimsLines.join("\n") : "<no claims>";

  const evidence = Array.isArray(output.evidence) ? output.evidence : [];
  const evLines = evidence.map((e, i) => {
    const txt = String(e.text || "").replace(/\\s+/g, " ");
    return `${i + 1}. [${e.chunk_id || ""}] (${e.doc_id || ""}) ${txt.slice(0, 240)}${txt.length > 240 ? "..." : ""}`;
  });
  document.getElementById("evidence_view").textContent = evLines.length ? evLines.join("\n") : "<no evidence>";

  document.getElementById("report_md").textContent = String(result.report_markdown || "");
  setSafetyBanner(result.safety_alert || null);
}

async function predictSeverity() {
  try {
    setStatus("Predicting severity...", "");
    const fileInput = document.getElementById("image_file");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("Please choose an image first.", "error");
      return;
    }
    const imageUpload = await fileToPayload(file);
    const payload = {
      image_upload: imageUpload,
      severity_predictor: document.getElementById("severity_predictor").value || "lookup"
    };
    const res = await fetch("/api/predict_severity", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const body = await res.json();
    if (!res.ok) {
      setStatus(body.error || "Severity prediction failed", "error");
      return;
    }
    document.getElementById("severity_json").value = JSON.stringify(body.severity || {}, null, 0);
    document.getElementById("predict_meta").textContent = `Predictor: ${body.predictor || "-"} | image: ${body.image_name || "-"}`;
    setStatus("Severity prediction loaded into severity JSON.", "ok");
  } catch (err) {
    setStatus("Error: " + err, "error");
  }
}

async function runWrapper() {
  try {
    setStatus("Running wrapper...", "");
    setSafetyBanner(null);

    const fileInput = document.getElementById("image_file");
    const file = fileInput.files && fileInput.files[0];
    let imageUpload = null;
    if (file) {
      imageUpload = await fileToPayload(file);
    }

    const payload = {
      query: document.getElementById("query").value,
      manifest_path: document.getElementById("manifest_path").value,
      mode: document.getElementById("mode").value,
      retrieval_k: Number(document.getElementById("retrieval_k").value || 5),
      retrieval_backend: document.getElementById("retrieval_backend").value || null,
      disable_rerank: document.getElementById("disable_rerank").value === "true",
      rerank_pool: Number(document.getElementById("rerank_pool").value || 20),
      rerank_alpha: Number(document.getElementById("rerank_alpha").value || 0.20),
      min_top_score_for_answer: Number(document.getElementById("min_top").value || 0.18),
      min_mean_score_for_answer: Number(document.getElementById("min_mean").value || 0.12),
      min_retrieved_for_answer: Number(document.getElementById("min_retrieved").value || 2),
      severity_json: document.getElementById("severity_json").value,
      severity_predictor: document.getElementById("severity_predictor").value || "lookup",
      image_upload: imageUpload
    };

    const res = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const body = await res.json();
    if (!res.ok) {
      setStatus(body.error || "Request failed", "error");
      return;
    }
    lastResult = body;
    renderRun(body);
    setStatus(`Done (${body.run_id})`, "ok");
    await loadHistory();
  } catch (err) {
    setStatus("Error: " + err, "error");
  }
}

function exportJson() {
  if (!lastResult) {
    setStatus("No run result available for export.", "error");
    return;
  }
  const runId = lastResult.run_id || "chapter5_ui_run";
  const payload = {
    run_id: runId,
    request: lastResult.request || {},
    info: lastResult.info || {},
    output: lastResult.output || {},
    safety_alert: lastResult.safety_alert || null,
    generated_utc: lastResult.generated_utc || null
  };
  downloadBlob(`${runId}_report.json`, JSON.stringify(payload, null, 2), "application/json");
}

function exportMarkdown() {
  if (!lastResult) {
    setStatus("No run result available for export.", "error");
    return;
  }
  const runId = lastResult.run_id || "chapter5_ui_run";
  const md = String(lastResult.report_markdown || "");
  downloadBlob(`${runId}_report.md`, md, "text/markdown");
}

async function loadHistory() {
  try {
    const res = await fetch("/api/history?limit=20");
    const body = await res.json();
    if (!res.ok) {
      return;
    }
    historyRows = Array.isArray(body.items) ? body.items.slice().reverse() : [];
    const host = document.getElementById("history_list");
    host.innerHTML = "";
    if (!historyRows.length) {
      host.innerHTML = '<p class="tiny">No history yet.</p>';
      return;
    }

    for (const row of historyRows) {
      const item = document.createElement("div");
      item.className = "history-item";
      const runId = row.run_id || "run";
      const query = String(row.request && row.request.query || "").slice(0, 120);
      const utc = row.generated_utc || "";
      item.innerHTML =
        `<div class="history-meta mono">${runId} | ${utc}</div>` +
        `<div>${query || "<no query>"}</div>`;
      item.addEventListener("click", () => {
        lastResult = row;
        renderRun(row);
        setStatus(`Loaded history run ${runId}`, "ok");
      });
      host.appendChild(item);
    }
  } catch (err) {
    // silent: history is convenience only
  }
}

document.getElementById("predict_btn").addEventListener("click", predictSeverity);
document.getElementById("run_btn").addEventListener("click", runWrapper);
document.getElementById("export_json_btn").addEventListener("click", exportJson);
document.getElementById("export_md_btn").addEventListener("click", exportMarkdown);
document.getElementById("refresh_history_btn").addEventListener("click", loadHistory);
loadHistory();
</script>
</body>
</html>
""".replace("__DEFAULT_MANIFEST_PATH__", default_manifest_path)


class UiServer:
    def __init__(
        self,
        *,
        manifest_path: Path,
        out_dir: Path,
        default_mode: str,
        default_k: int,
        severity_lookup_csv: Optional[Path],
        severity_lookup_run_id: str,
        severity_predict_cmd: Optional[str],
        predict_timeout_sec: int,
    ) -> None:
        self.manifest_path = manifest_path
        self.out_dir = out_dir
        self.default_mode = default_mode
        self.default_k = default_k
        self.severity_lookup_csv = severity_lookup_csv
        self.severity_lookup_run_id = severity_lookup_run_id
        self.severity_predict_cmd = severity_predict_cmd
        self.predict_timeout_sec = predict_timeout_sec
        self.lookup_predictions = _load_lookup_predictions(severity_lookup_csv, severity_lookup_run_id)

    def make_handler(self):
        manifest_path = self.manifest_path
        out_dir = self.out_dir
        default_mode = self.default_mode
        default_k = self.default_k
        severity_lookup_csv = self.severity_lookup_csv
        severity_lookup_run_id = self.severity_lookup_run_id
        severity_predict_cmd = self.severity_predict_cmd
        predict_timeout_sec = self.predict_timeout_sec
        lookup_predictions = self.lookup_predictions

        workspace_root = _find_workspace_root()
        import sys

        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from pico_wrapper.schemas import SeverityResult
        from pico_wrapper.ui_support import build_markdown_report, build_safety_alert
        from pico_wrapper.utils_io import generate_run_id, write_json
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

            def _read_json(self) -> Dict[str, Any]:
                cl = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(cl).decode("utf-8") if cl > 0 else ""
                req = json.loads(raw) if raw else {}
                if not isinstance(req, dict):
                    raise ValueError("request body must be JSON object")
                return req

            def _predict_severity(
                self,
                *,
                req: Mapping[str, Any],
                image_upload: Mapping[str, Any],
                run_id: str,
            ) -> tuple[Dict[str, Any], Dict[str, Any]]:
                blob, original_name, _mime = _decode_image_upload(image_upload)
                saved_path = _save_upload(out_dir=out_dir, run_id=run_id, filename=original_name, blob=blob)

                predictor = str(req.get("severity_predictor", "lookup")).strip().lower()
                if predictor not in {"lookup", "command"}:
                    predictor = "lookup"

                severity: Optional[Dict[str, Any]] = None
                details: Dict[str, Any] = {
                    "predictor": predictor,
                    "image_name": original_name,
                    "image_path": str(saved_path),
                    "severity_lookup_csv": None if severity_lookup_csv is None else str(severity_lookup_csv),
                }

                if predictor == "lookup":
                    severity = _find_lookup_severity(
                        lookup_predictions,
                        image_path=saved_path,
                        original_filename=original_name,
                    )
                    if severity is None:
                        raise RuntimeError(
                            "lookup predictor could not match this image name against loaded Chapter 4 prediction CSV"
                        )
                else:
                    cmd = str(req.get("severity_predict_cmd", "")).strip() or (severity_predict_cmd or "")
                    if not cmd:
                        raise RuntimeError(
                            "severity_predict_cmd is required for command predictor mode"
                        )
                    severity = _predict_severity_with_command(
                        cmd_template=cmd,
                        image_path=saved_path,
                        workspace_root=workspace_root.parents[1],
                        timeout_sec=_as_int(req.get("predict_timeout_sec"), default=predict_timeout_sec),
                    )

                validated = SeverityResult.from_dict(severity).to_dict()
                details["severity"] = validated
                return validated, details

            def _parse_severity_payload(self, req: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
                severity_json = req.get("severity_json")
                if severity_json is None:
                    return None
                if isinstance(severity_json, str):
                    raw = severity_json.strip()
                    if not raw:
                        return None
                    obj = json.loads(raw)
                elif isinstance(severity_json, dict):
                    obj = severity_json
                else:
                    raise ValueError("severity_json must be string or object")
                return SeverityResult.from_dict(obj).to_dict()

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/" or parsed.path == "":
                    self._html_response(_build_html(str(manifest_path)))
                    return
                if parsed.path == "/api/history":
                    qs = parse_qs(parsed.query)
                    limit = _as_int((qs.get("limit") or [20])[0], default=20)
                    sessions = _read_jsonl_tail(out_dir / "ui_sessions.jsonl", limit=max(1, min(200, limit)))
                    self._json_response(HTTPStatus.OK, {"items": sessions})
                    return
                if parsed.path == "/api/health":
                    self._json_response(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "manifest_path": str(manifest_path),
                            "lookup_rows": len(lookup_predictions),
                            "has_predict_cmd": bool(severity_predict_cmd),
                        },
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                try:
                    if parsed.path == "/api/predict_severity":
                        req = self._read_json()
                        upload = req.get("image_upload")
                        if not isinstance(upload, dict):
                            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "image_upload is required"})
                            return
                        run_id = generate_run_id("chapter5_ui_pred")
                        severity, details = self._predict_severity(req=req, image_upload=upload, run_id=run_id)
                        _append_jsonl(
                            out_dir / "ui_requests.jsonl",
                            {
                                "ts_utc": _utc_now_iso(),
                                "request_type": "predict_severity",
                                "run_id": run_id,
                                "predictor": details.get("predictor"),
                                "image_name": details.get("image_name"),
                                "image_path": details.get("image_path"),
                                "ok": True,
                            },
                        )
                        self._json_response(
                            HTTPStatus.OK,
                            {
                                "run_id": run_id,
                                "predictor": details.get("predictor"),
                                "image_name": details.get("image_name"),
                                "image_path": details.get("image_path"),
                                "severity": severity,
                            },
                        )
                        return

                    if parsed.path != "/api/run":
                        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                        return

                    req = self._read_json()
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

                    retrieval_k = _as_int(req.get("retrieval_k"), default=default_k)
                    if retrieval_k < 1:
                        retrieval_k = default_k

                    retrieval_backend = req.get("retrieval_backend")
                    if retrieval_backend is not None:
                        retrieval_backend = str(retrieval_backend).strip().lower() or None
                    if retrieval_backend not in {None, "keyword", "tfidf", "hybrid"}:
                        retrieval_backend = None

                    disable_rerank = _as_bool(req.get("disable_rerank"), default=False)
                    rerank_pool = max(1, _as_int(req.get("rerank_pool"), default=20))
                    rerank_alpha = _as_float(req.get("rerank_alpha"), default=0.20)

                    min_top = max(0.0, _as_float(req.get("min_top_score_for_answer"), default=0.18))
                    min_mean = max(0.0, _as_float(req.get("min_mean_score_for_answer"), default=0.12))
                    min_retrieved = max(1, _as_int(req.get("min_retrieved_for_answer"), default=2))

                    severity = self._parse_severity_payload(req)
                    predicted_details: Optional[Dict[str, Any]] = None
                    image_upload = req.get("image_upload")
                    if severity is None and isinstance(image_upload, dict):
                        pred_run_id = generate_run_id("chapter5_ui_pred")
                        severity, predicted_details = self._predict_severity(
                            req=req,
                            image_upload=image_upload,
                            run_id=pred_run_id,
                        )

                    run_id = generate_run_id("chapter5_ui")
                    output, info = run_wrapper(
                        query=query,
                        manifest_path=req_manifest,
                        run_id=run_id,
                        retrieval_k=retrieval_k,
                        retrieval_backend=retrieval_backend,
                        enable_rerank=(not disable_rerank),
                        rerank_pool=rerank_pool,
                        rerank_alpha=rerank_alpha,
                        min_top_score_for_answer=min_top,
                        min_mean_score_for_answer=min_mean,
                        min_retrieved_for_answer=min_retrieved,
                        mode=mode,
                        severity=severity,
                    )

                    info_dict = info.to_dict()
                    output_dict = output.to_dict()
                    safety_alert = build_safety_alert(run_info=info_dict, wrapper_output=output_dict)
                    request_payload = {
                        "query": query,
                        "manifest_path": str(req_manifest),
                        "mode": mode,
                        "retrieval_k": retrieval_k,
                        "retrieval_backend": retrieval_backend,
                        "disable_rerank": disable_rerank,
                        "rerank_pool": rerank_pool,
                        "rerank_alpha": rerank_alpha,
                        "min_top_score_for_answer": min_top,
                        "min_mean_score_for_answer": min_mean,
                        "min_retrieved_for_answer": min_retrieved,
                        "severity_predictor": req.get("severity_predictor"),
                        "has_severity_json": severity is not None,
                    }
                    generated_utc = _utc_now_iso()
                    report_markdown = build_markdown_report(
                        run_id=run_id,
                        request_payload=request_payload,
                        run_info=info_dict,
                        wrapper_output=output_dict,
                        safety_alert=safety_alert,
                        generated_utc=generated_utc,
                    )

                    report_dir = out_dir / "reports"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_json_path = report_dir / f"{run_id}_report.json"
                    report_md_path = report_dir / f"{run_id}_report.md"
                    report_payload = {
                        "run_id": run_id,
                        "generated_utc": generated_utc,
                        "request": request_payload,
                        "info": info_dict,
                        "output": output_dict,
                        "safety_alert": safety_alert,
                        "predicted_severity": predicted_details,
                    }
                    write_json(report_json_path, report_payload)
                    report_md_path.write_text(report_markdown, encoding="utf-8")

                    _append_jsonl(out_dir / "wrapper_outputs.jsonl", output_dict)
                    _append_jsonl(out_dir / "wrapper_run_infos.jsonl", info_dict)
                    _append_jsonl(
                        out_dir / "ui_requests.jsonl",
                        {
                            "ts_utc": generated_utc,
                            "request_type": "run",
                            "run_id": run_id,
                            **request_payload,
                        },
                    )

                    session_row = {
                        "run_id": run_id,
                        "generated_utc": generated_utc,
                        "request": request_payload,
                        "info": info_dict,
                        "output": output_dict,
                        "safety_alert": safety_alert,
                        "report_markdown": report_markdown,
                        "report_json_path": str(report_json_path),
                        "report_md_path": str(report_md_path),
                    }
                    _append_jsonl(out_dir / "ui_sessions.jsonl", session_row)

                    self._json_response(
                        HTTPStatus.OK,
                        {
                            "run_id": run_id,
                            "generated_utc": generated_utc,
                            "request": request_payload,
                            "info": info_dict,
                            "output": output_dict,
                            "safety_alert": safety_alert,
                            "report_markdown": report_markdown,
                            "report_json_path": str(report_json_path),
                            "report_md_path": str(report_md_path),
                            "predicted_severity": predicted_details,
                        },
                    )
                except json.JSONDecodeError as exc:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
                except Exception as exc:  # noqa: BLE001
                    _append_jsonl(
                        out_dir / "ui_errors.jsonl",
                        {
                            "ts_utc": _utc_now_iso(),
                            "path": parsed.path,
                            "error": str(exc),
                            "trace": traceback.format_exc(),
                        },
                    )
                    self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
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
    parser.add_argument(
        "--severity_lookup_csv",
        type=Path,
        default=_default_lookup_csv(),
        help="Optional Chapter 4 prediction CSV used by lookup severity predictor.",
    )
    parser.add_argument(
        "--severity_lookup_run_id",
        type=str,
        default="finetune_resnet50",
        help="Run id embedded in lookup-derived severity payloads.",
    )
    parser.add_argument(
        "--severity_predict_cmd",
        type=str,
        default=None,
        help=(
            "Optional external command template for severity prediction. "
            "Must emit JSON for SeverityResult and include '{image_path}' placeholder."
        ),
    )
    parser.add_argument("--predict_timeout_sec", type=int, default=120)
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

    severity_lookup_csv = args.severity_lookup_csv.resolve() if args.severity_lookup_csv is not None else None
    if severity_lookup_csv is not None and not severity_lookup_csv.exists():
        print(f"[warn] severity lookup CSV not found, lookup predictor disabled: {severity_lookup_csv}")
        severity_lookup_csv = None

    ui = UiServer(
        manifest_path=manifest_path,
        out_dir=out_dir,
        default_mode=args.mode,
        default_k=args.retrieval_k,
        severity_lookup_csv=severity_lookup_csv,
        severity_lookup_run_id=args.severity_lookup_run_id,
        severity_predict_cmd=args.severity_predict_cmd,
        predict_timeout_sec=args.predict_timeout_sec,
    )
    handler = ui.make_handler()
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Chapter 5 UI running at http://{args.host}:{args.port}")
    print(f"Default manifest: {manifest_path}")
    print(f"Output log dir: {out_dir}")
    print(f"Lookup predictor rows: {len(ui.lookup_predictions)}")
    print(f"Lookup CSV: {severity_lookup_csv}")
    print(f"Command predictor configured: {bool(args.severity_predict_cmd)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
