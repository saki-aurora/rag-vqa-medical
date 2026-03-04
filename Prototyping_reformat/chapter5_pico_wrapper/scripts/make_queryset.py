#!/usr/bin/env python3
"""Generate synthetic physician-style query sets and gold labels for Part 4."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def _build_queries(n_queries: int = 50) -> List[Dict[str, object]]:
    populations = [
        "adults with ulcerative colitis",
        "adults with moderate ulcerative colitis",
        "adults with severe ulcerative colitis",
        "outpatients with ulcerative colitis",
        "patients with MES 2 ulcerative colitis",
    ]
    interventions = [
        "biologic therapy",
        "infliximab",
        "vedolizumab",
        "ustekinumab",
        "advanced therapy",
    ]
    comparators = [
        "placebo",
        "standard care",
        "usual care",
        "adalimumab",
        "tofacitinib",
    ]
    outcomes = [
        "clinical remission",
        "endoscopic improvement",
        "steroid-free remission",
        "adverse events",
        "hospitalization risk",
    ]
    timeframes = ["8 weeks", "12 weeks", "24 weeks", "short-term", "long-term"]
    settings = ["outpatient", "inpatient", "clinic", "hospital"]

    templates = [
        "In {pop}, does {intv} versus {comp} improve {out} at {tf} in {setg} setting?",
        "For {pop}, compare {intv} with {comp} for {out} over {tf}.",
        "What is the evidence that {intv} versus {comp} changes {out} in {pop} at {tf}?",
        "In {setg} care, should {pop} receive {intv} rather than {comp} for {out} at {tf}?",
    ]

    out: List[Dict[str, object]] = []
    for i in range(n_queries):
        pop = populations[i % len(populations)]
        intv = interventions[i % len(interventions)]
        comp = comparators[i % len(comparators)]
        outc = outcomes[i % len(outcomes)]
        tf = timeframes[i % len(timeframes)]
        setg = settings[i % len(settings)]
        tmpl = templates[i % len(templates)]

        q = tmpl.format(pop=pop, intv=intv, comp=comp, out=outc, tf=tf, setg=setg)

        # Inject severity anchors regularly.
        if i % 3 == 0:
            q += f" The patient currently has MES {i % 4}."
        if i % 7 == 0:
            q += " Consider UCEIS 5 context."

        # Inject a few explicit safety-sensitive queries.
        if i in {11, 23, 37, 49}:
            q = q + " Also, what dose in mg should be prescribed?"

        out.append(
            {
                "qid": f"q{i+1:03d}",
                "query": q,
                "source": "synthetic_part4",
            }
        )
    return out


def _build_pico_gold(queries: List[Dict[str, object]], n_gold: int = 20) -> List[Dict[str, object]]:
    """Hand-crafted gold subset for extraction evaluation."""
    gold: List[Dict[str, object]] = []
    for row in queries[:n_gold]:
        qid = row["qid"]
        q = row["query"]
        # Conservative gold labels aligned with current rule-based extractor vocabulary.
        pop = []
        if "adults" in q:
            pop.append("adults")
        if "ulcerative colitis" in q:
            pop.append("ulcerative colitis")
        if "moderate" in q:
            pop.append("moderate")
        if "severe" in q:
            pop.append("severe")

        intervention = []
        for term in ["biologic", "infliximab", "vedolizumab", "ustekinumab", "advanced therapy"]:
            if term in q.lower():
                intervention.append(term)

        comparator = []
        for term in ["placebo", "standard care", "usual care", "versus", "vs"]:
            if term in q.lower():
                comparator.append(term)

        outcomes = []
        for term in ["clinical remission", "endoscopic improvement", "steroid-free remission", "adverse events"]:
            if term in q.lower():
                outcomes.append(term)
        if not outcomes and "remission" in q.lower():
            outcomes.append("remission")

        severity_anchors = []
        if "MES 0" in q:
            severity_anchors.append("MES 0")
        if "MES 1" in q:
            severity_anchors.append("MES 1")
        if "MES 2" in q:
            severity_anchors.append("MES 2")
        if "MES 3" in q:
            severity_anchors.append("MES 3")
        if "UCEIS 5" in q:
            severity_anchors.append("UCEIS 5")

        timeframe = None
        for tf in ["8 weeks", "12 weeks", "24 weeks", "short-term", "long-term"]:
            if tf in q:
                timeframe = tf
                break

        setting = None
        for s in ["outpatient", "inpatient", "clinic", "hospital"]:
            if s in q.lower():
                setting = s
                break

        constraints = []
        if "dose in mg" in q.lower():
            constraints.append("dose in mg")

        gold.append(
            {
                "qid": qid,
                "query": q,
                "pico": {
                    "population": pop,
                    "intervention": intervention,
                    "comparator": comparator,
                    "outcomes": outcomes,
                    "severity_anchors": severity_anchors,
                    "timeframe": timeframe,
                    "setting": setting,
                    "constraints": constraints,
                },
            }
        )
    return gold


def _build_retrieval_gold(queries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Create retrieval labels for 10 queries using sample KB chunk IDs."""
    labels: List[Dict[str, object]] = []
    mapping = [
        ("q001", ["sample_docs__uc_biologics_notes#c0002", "sample_docs__uc_biologics_notes#c0004"]),
        ("q002", ["sample_docs__uc_biologics_notes#c0003", "sample_docs__uc_biologics_notes#c0004"]),
        ("q003", ["sample_docs__uc_biologics_notes#c0003", "sample_docs__uc_biologics_notes#c0002"]),
        ("q004", ["sample_docs__uc_biologics_notes#c0004"]),
        ("q005", ["sample_docs__uc_guideline_summary#c0003", "sample_docs__uc_guideline_summary#c0004"]),
        ("q006", ["sample_docs__uc_guideline_summary#c0003"]),
        ("q007", ["sample_docs__uc_guideline_summary#c0003", "sample_docs__uc_followup_and_uncertainty#c0002"]),
        ("q008", ["sample_docs__uc_followup_and_uncertainty#c0002"]),
        ("q009", ["sample_docs__uc_followup_and_uncertainty#c0003"]),
        ("q010", ["sample_docs__uc_followup_and_uncertainty#c0004"]),
    ]
    qmap = {row["qid"]: row for row in queries}
    for qid, chunk_ids in mapping:
        q = qmap[qid]["query"]
        doc_ids = sorted({cid.split("#")[0] for cid in chunk_ids})
        labels.append(
            {
                "qid": qid,
                "query": q,
                "relevant_chunk_ids": chunk_ids,
                "relevant_doc_ids": doc_ids,
            }
        )
    return labels


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            import json

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    root = _find_workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=root / "data" / "queries")
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--n_pico_gold", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = _build_queries(n_queries=args.n_queries)
    pico_gold = _build_pico_gold(queries=queries, n_gold=args.n_pico_gold)
    retrieval_gold = _build_retrieval_gold(queries=queries)

    _write_jsonl(args.out_dir / "queries.jsonl", queries)
    _write_jsonl(args.out_dir / "pico_gold.jsonl", pico_gold)
    _write_jsonl(args.out_dir / "retrieval_gold.jsonl", retrieval_gold)

    print(f"Wrote {len(queries)} queries -> {args.out_dir / 'queries.jsonl'}")
    print(f"Wrote {len(pico_gold)} pico gold rows -> {args.out_dir / 'pico_gold.jsonl'}")
    print(f"Wrote {len(retrieval_gold)} retrieval gold rows -> {args.out_dir / 'retrieval_gold.jsonl'}")


if __name__ == "__main__":
    main()
