# Chapter 5 Figure Sync (Part 5)

Date: 2026-03-06

## Scope

Synchronized Chapter 5 figures (`F07` to `F10`) and source data to the frozen pass4 artifact set used by Chapter 5 reporting.

## Inputs

- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass3_ablation/retrieval_ablation_summary.tsv`
- `Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_pass4_latest/chapter5_completion_report.json`

## Pipeline Source

- `Prototyping_reformat/thesis_figures/part3_ch5_figures.py`
- `Prototyping_reformat/thesis_figures/out/part3_ch5_summary.md`

## Chapter 5 Figures (Staged)

- `Thesis/markdown/figures/generated/F07_ch5_pico_field_precision_recall_f1.png`
- `Thesis/markdown/figures/generated/F08_ch5_retrieval_at_k_curve_with_ci.png`
- `Thesis/markdown/figures/generated/F09_ch5_retrieval_ablation_comparison.png`
- `Thesis/markdown/figures/generated/F10_ch5_answer_quality_grounding_kpi_panel.png`

## Supporting Data Exports

- `Prototyping_reformat/thesis_figures/data/ch5_pico_field_metrics.csv`
- `Prototyping_reformat/thesis_figures/data/ch5_pico_per_query_scores.csv`
- `Prototyping_reformat/thesis_figures/data/ch5_retrieval_curve.csv`
- `Prototyping_reformat/thesis_figures/data/ch5_retrieval_ablation.csv`
- `Prototyping_reformat/thesis_figures/data/ch5_answer_kpis.csv`
- `Prototyping_reformat/thesis_figures/data/ch5_answer_counts.csv`
- `Prototyping_reformat/thesis_figures/data/ch5_completion_audit_checklist.csv`

## Result

Chapter 5 figures and figure-source tables are synchronized to the frozen pass4 artifact boundary and ready for dissertation insertion/citation.
