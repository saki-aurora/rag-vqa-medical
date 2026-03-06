"""Helpers for UI-facing report/export and safety summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _as_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def build_safety_alert(
    *,
    run_info: Mapping[str, Any],
    wrapper_output: Mapping[str, Any],
) -> Optional[Dict[str, str]]:
    refusal = _as_bool(wrapper_output.get("refusal")) or _as_bool(run_info.get("refusal_triggered"))
    abstained = _as_bool(run_info.get("abstained_low_evidence"))
    abstain_reason = _as_str(run_info.get("abstain_reason"))
    limitations = [_as_str(x).lower() for x in _safe_list(wrapper_output.get("limitations"))]

    if refusal:
        return {
            "level": "danger",
            "title": "Escalation Required",
            "message": (
                "High-risk request detected (for example dosing/emergency language). "
                "Use emergency protocol and clinician escalation."
            ),
        }

    if abstained:
        msg = "Low-evidence abstention triggered."
        if abstain_reason:
            msg += f" Rule: {abstain_reason}."
        return {
            "level": "warning",
            "title": "Low-Evidence Warning",
            "message": msg,
        }

    if any(
        ("contraindication" in item) or ("adverse-event" in item) or ("adverse event" in item)
        for item in limitations
    ):
        return {
            "level": "warning",
            "title": "Contraindication Caution",
            "message": "Contraindication/adverse-event context detected. Verify with guidelines before action.",
        }

    return None


def _iter_claim_lines(claims: Iterable[object]) -> list[str]:
    out: list[str] = []
    for idx, row in enumerate(claims, start=1):
        if not isinstance(row, Mapping):
            continue
        text = _as_str(row.get("text"))
        cids = row.get("citation_ids")
        cid_text = ", ".join(str(x) for x in cids) if isinstance(cids, list) else ""
        if cid_text:
            out.append(f"{idx}. {text} (citations: {cid_text})")
        else:
            out.append(f"{idx}. {text}")
    return out


def _iter_evidence_lines(evidence: Iterable[object], max_chars: int = 220) -> list[str]:
    out: list[str] = []
    for idx, row in enumerate(evidence, start=1):
        if not isinstance(row, Mapping):
            continue
        cid = _as_str(row.get("chunk_id"))
        doc_id = _as_str(row.get("doc_id"))
        text = _as_str(row.get("text")).replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        out.append(f"{idx}. [{cid}] ({doc_id}) {text}")
    return out


def build_markdown_report(
    *,
    run_id: str,
    request_payload: Mapping[str, Any],
    run_info: Mapping[str, Any],
    wrapper_output: Mapping[str, Any],
    safety_alert: Optional[Mapping[str, str]],
    generated_utc: Optional[str] = None,
) -> str:
    ts = generated_utc or utc_now_iso()
    query = _as_str(request_payload.get("query"))
    mode = _as_str(request_payload.get("mode")) or _as_str(run_info.get("used_mode"))
    retrieval_k = run_info.get("retrieval_k", request_payload.get("retrieval_k", 5))
    backend = _as_str(run_info.get("retrieval_backend"))

    lines: list[str] = []
    lines.append(f"# Chapter 5 Wrapper Report: `{run_id}`")
    lines.append("")
    lines.append(f"- Generated UTC: `{ts}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Retrieval K: `{retrieval_k}`")
    if backend:
        lines.append(f"- Retrieval Backend: `{backend}`")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append(query or "<empty>")
    lines.append("")

    if safety_alert:
        lines.append("## Safety Alert")
        lines.append("")
        lines.append(f"- Level: `{_as_str(safety_alert.get('level'))}`")
        lines.append(f"- Title: {_as_str(safety_alert.get('title'))}")
        lines.append(f"- Message: {_as_str(safety_alert.get('message'))}")
        lines.append("")

    pico = wrapper_output.get("pico")
    if isinstance(pico, Mapping):
        lines.append("## PICO")
        lines.append("")
        for key in [
            "population",
            "intervention",
            "comparator",
            "outcomes",
            "severity_anchors",
            "timeframe",
            "setting",
            "constraints",
        ]:
            value = pico.get(key)
            if isinstance(value, list):
                txt = ", ".join(str(x) for x in value) if value else "-"
            else:
                txt = _as_str(value) or "-"
            lines.append(f"- {key}: {txt}")
        lines.append("")

    claim_lines = _iter_claim_lines(_safe_list(wrapper_output.get("claims")))
    lines.append("## Claims")
    lines.append("")
    if claim_lines:
        lines.extend(claim_lines)
    else:
        lines.append("- No claims produced.")
    lines.append("")

    evidence_lines = _iter_evidence_lines(_safe_list(wrapper_output.get("evidence")))
    lines.append("## Evidence (Top Retrieved)")
    lines.append("")
    if evidence_lines:
        lines.extend(evidence_lines)
    else:
        lines.append("- No evidence retrieved.")
    lines.append("")

    uncertainty = _as_str(wrapper_output.get("uncertainty"))
    lines.append("## Uncertainty")
    lines.append("")
    lines.append(uncertainty or "-")
    lines.append("")

    limits = _safe_list(wrapper_output.get("limitations"))
    lines.append("## Limitations")
    lines.append("")
    if limits:
        for item in limits:
            lines.append(f"- {_as_str(item)}")
    else:
        lines.append("- None listed.")
    lines.append("")

    disclaimer = _as_str(wrapper_output.get("disclaimer"))
    lines.append("## Disclaimer")
    lines.append("")
    lines.append(disclaimer or "-")
    lines.append("")

    return "\n".join(lines).strip() + "\n"
