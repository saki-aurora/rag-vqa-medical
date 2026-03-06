# Chapter 4 Scope Freeze (Part 1)

Date: 2026-03-06
Scope owner: dissertation drafting freeze for Chapter 4

## 1) Official KPI Lock

Primary KPI (optimization + claim focus):
- Internal LIMUC `mode1/test` Quadratic Weighted Kappa (QWK)

Secondary KPIs (reported with primary):
- Accuracy
- Macro-F1
- Balanced accuracy
- Parse rate (generative lanes)

External policy:
- HyperKvasir UC proxy is a stress-test only.
- External metrics are reported as domain-shift evidence, not as primary optimization target.

## 2) Official Results to Report (Headline)

These are the frozen, official Chapter 4 numbers for dissertation claims.

### 2.1 Supervised baseline (official)
Source: `pass5_supervised_metric_summary.csv`
- Run family: Pass 5 ResNet50 supervised, multi-seed (11/23/42)
- Accuracy: `0.737643`
- Macro-F1: `0.667330`
- Balanced accuracy: `0.670907`
- QWK: `0.818649`
- 95% CI QWK: `[0.807920, 0.830582]`

### 2.2 Generative primary result (official)
Source: `pass6_generative_metric_summary.csv` (lane=`lora_mode1_train`)
- Run family: Pass 6 LoRA mode1, multi-seed (11/23/77)
- Accuracy: `0.781930`
- Macro-F1: `0.727920`
- Balanced accuracy: `0.736292`
- QWK: `0.863656`
- Parse rate: `1.000000`
- 95% CI QWK: `[0.862382, 0.865836]`

### 2.3 Controlled generative ablation (official ablation)
Source: `pass6_generative_metric_summary.csv` (lane=`lora_mode2_eval`)
- Accuracy: `0.548636`
- Macro-F1: `0.177135`
- Balanced accuracy: `0.250000`
- QWK: `0.000000`
- Parse rate: `1.000000`

### 2.4 External stress-test (official limitation evidence)
Source: `pass7_external_validation_report.json`

Internal reference (for drop context):
- `resnet50_supervised`: acc `0.763345`, macro-F1 `0.669402`, QWK `0.828762`
- `vlm_lora_mode1`: acc `0.781139`, macro-F1 `0.720147`, QWK `0.862752`, parse rate `1.0`

External HyperKvasir UC proxy:
- `resnet50_supervised`: acc `0.433608`, macro-F1 `0.387423`, QWK `0.359597`
- `vlm_lora_mode1`: acc `0.041128`, macro-F1 `0.019752`, QWK `0.000000`, parse rate `0.0`
- `vlm_lora_mode2`: acc `0.041128`, macro-F1 `0.019752`, QWK `0.000000`, parse rate `1.0`

## 3) Exploratory (Non-Headline) Results

These are not headline dissertation claims; include only as "exploratory optimization" if needed.

### 3.1 Pass 8 fusion experiments
Source: `pass8_internal_fusion_20260306T085817Z/pass8_internal_fusion_candidates.csv`
- Best test QWK observed in this sweep: `0.866368` (`baseline_vlm_vote3`)
- Several high-val models overfit and did not improve test beyond the frozen mode1 range.

### 3.2 5090 supervised push (scout/focus)
Sources:
- `pass8_supervised_5090_scout_r2_20260306T091539Z/`
- `pass8_supervised_5090_focus_20260306T092204Z/`

Best single supervised run observed:
- `swin_t_ce_m_e8_seed011`: acc `0.788256`, macro-F1 `0.732254`, QWK `0.870416`

Follow-up seeds on same family:
- `swin_t_ce_m_e8_seed023`: QWK `0.866927`
- `swin_t_ce_m_e8_seed077`: QWK `0.868124`

Interpretation:
- Supervised improvements were real but still below the target `QWK >= 0.90`.
- Keep this as exploratory optimization evidence, not core claimed performance.

## 4) Final Claim Boundary (for writing)

Headline claims allowed:
1. On internal LIMUC, multi-seed LoRA mode1 outperforms the frozen supervised multi-seed baseline in QWK and macro-F1.
2. Controlled mode2 scoring collapses for this setup and is retained as an ablation/failure case.
3. External HyperKvasir proxy performance drops sharply, supporting a domain-shift and label-mapping mismatch limitation.

Claims to avoid in headline:
1. Do not claim internal `QWK >= 0.90` was achieved.
2. Do not claim external generalization readiness.
3. Do not mix exploratory Pass 8 numbers into frozen primary tables.

## 5) Future-Run Bucket (deferred)

Deferred for future experiments (not required to start dissertation writing):
- Data cleanup/relabel pass focused on class-boundary ambiguity (`0<->1`, `1<->2`).
- Re-run top supervised/generative pipelines on cleaned metadata.
- Acquire and evaluate a true Mayo-compatible external dataset with native 0/1/2/3 labels.

## 6) Frozen Artifact Paths

- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_internal_fusion_20260306T085817Z/pass8_internal_fusion_candidates.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_scout_r2_20260306T091539Z/pass8_supervised_scout_runs.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_focus_20260306T092204Z/pass8_supervised_all_runs.csv`

