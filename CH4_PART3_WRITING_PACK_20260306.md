# Chapter 4 Writing Pack (Part 3)

Date: 2026-03-06
Input freezes consumed:
- `CH4_PART1_SCOPE_FREEZE_20260306.md`
- `CH4_PART2_REPRO_FREEZE_20260306.md`

## 1) Purpose

This pack converts the frozen Chapter 4 evidence into a writing-ready asset map and citation-ready metric blocks.

Machine-readable manifest:
- `CH4_PART3_ASSET_MANIFEST_20260306.csv`

## 2) Frozen Numbers to Cite (Primary Claims)

Use these exact values for Chapter 4 headline claims.

### 2.1 Supervised baseline (Pass 5, multi-seed)
Source: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_metric_summary.csv`

- Accuracy: `0.737643`
- Macro-F1: `0.667330`
- Balanced accuracy: `0.670907`
- QWK: `0.818649`
- 95% CI QWK: `[0.807920, 0.830582]`

### 2.2 Generative primary (Pass 6 mode1, multi-seed)
Source: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`

- Accuracy: `0.781930`
- Macro-F1: `0.727920`
- Balanced accuracy: `0.736292`
- QWK: `0.863656`
- Parse rate: `1.000000`
- 95% CI QWK: `[0.862382, 0.865836]`

### 2.3 Controlled ablation (Pass 6 mode2)
Source: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`

- Accuracy: `0.548636`
- Macro-F1: `0.177135`
- Balanced accuracy: `0.250000`
- QWK: `0.000000`
- Parse rate: `1.000000`

### 2.4 External stress-test (Pass 7 HyperKvasir UC proxy)
Source: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`

- ResNet50 supervised QWK: internal `0.828762` -> external `0.359597` (delta `-0.469165`)
- VLM mode1 QWK: internal `0.862752` -> external `0.000000` (delta `-0.862752`)
- VLM mode1 parse rate: internal `1.0` -> external `0.0`

## 3) Table Order for Chapter 4 Draft

Recommended insertion order (all paths already in `CH4_PART3_ASSET_MANIFEST_20260306.csv`):

1. `T4_01` supervised baseline summary (Pass 5)
2. `T4_02` generative primary + controlled ablation (Pass 6)
3. `T4_03` mode1 QC pass table (seed stability)
4. `T4_04` external drop table (Pass 7)
5. `T4_05` external detailed report reference (Pass 7 JSON)

Appendix-only:
- `T4_A1`, `T4_A2`, `T4_A3` (Pass 8 exploratory)

## 4) Figure Order for Chapter 4 Draft

Use these official report figures (not legacy thesis-pack figures):

1. `F4_01` pass5 metric CI: `.../out/figures/pass5_supervised_metric_ci.png`
2. `F4_02` pass6 metric CI: `.../out/figures/pass6_generative_metric_ci.png`
3. `F4_03` pass5 aggregate confusion: `.../out/pass5_supervised_confusion_aggregate.png`
4. `F4_04` pass6 mode1 aggregate confusion: `.../out/pass6_generative_confusion_aggregate_mode1.png`
5. `F4_05` pass6 mode2 aggregate confusion: `.../out/pass6_generative_confusion_aggregate_mode2.png`
6. `F4_06` generative distribution: `.../out/generative_pred_distribution.png`

Important:
- `F4_L1` to `F4_L5` in the manifest are marked `legacy` and should not be used for new headline claims unless regenerated against Pass 5/6 freeze.

## 5) Ready-to-Paste Claim Blocks

### Block A: Internal primary result

"On internal LIMUC, the frozen multi-seed generative mode1 configuration (Pass 6) achieved higher agreement and class-balanced performance than the frozen supervised baseline aggregate (Pass 5), reaching QWK 0.8637 versus 0.8186, with macro-F1 0.7279 versus 0.6673."

### Block B: Controlled ablation result

"In controlled mode2 label scoring, performance collapsed (QWK 0.0000; macro-F1 0.1771), supporting the decision to keep mode1 as the primary generative lane for Chapter 4 claims."

### Block C: External limitation statement

"On the HyperKvasir UC proxy stress test, both lanes showed substantial degradation (e.g., VLM mode1 internal QWK 0.8628 to external 0.0000), indicating strong domain-shift and label-mapping mismatch; therefore external robustness is reported as a limitation rather than a primary optimization outcome."

## 6) What Is Ready vs Deferred

Ready for dissertation writing now:
- Scope freeze (Part 1)
- Repro/code freeze (Part 2)
- Writing asset map and metric blocks (Part 3, this file)

Deferred (future runs only):
- New training attempts for `QWK >= 0.90`
- External dataset replacement with native Mayo-compatible labels
- Regeneration of thesis-pack Chapter 4 legacy figures (`F4_L1`..`F4_L5`) against Pass 5/6 freeze

## 7) Verification

Quick check command:

```bash
cd /mnt/hf/thesis/rag-vqa-medical
python - <<'PY'
import pandas as pd
m=pd.read_csv('CH4_PART3_ASSET_MANIFEST_20260306.csv')
print(m[['asset_id','asset_type','status','exists']].to_string(index=False))
PY
```
