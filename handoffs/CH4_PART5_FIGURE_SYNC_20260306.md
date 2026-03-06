# Chapter 4 Figure Sync (Part 5)

Date: 2026-03-06

## Scope

Regenerated and restaged thesis Chapter 4 figures (`F02` to `F06`) to align with frozen Chapter 4 evidence (Pass 5/6/7), and refreshed figure manifest/catalog integration.

## Code Updates

- Updated canonical Chapter 4 figure inputs in:
  - `Prototyping_reformat/thesis_figures/freeze_inputs.py`
- Replaced Chapter 4 figure generation logic with frozen pass-based pipeline in:
  - `Prototyping_reformat/thesis_figures/part2_ch4_figures.py`
- Updated Chapter 4 figure captions in:
  - `Prototyping_reformat/thesis_figures/part5_thesis_integration.py`
- Updated workspace notes in:
  - `Prototyping_reformat/thesis_figures/README.md`

## Pipeline Run

Executed:

```bash
python Prototyping_reformat/thesis_figures/freeze_inputs.py
python Prototyping_reformat/thesis_figures/part2_ch4_figures.py
python Prototyping_reformat/thesis_figures/part3_ch5_figures.py
python Prototyping_reformat/thesis_figures/part4_ch3_figures.py
python Prototyping_reformat/thesis_figures/part5_thesis_integration.py
```

Final integration status:
- `figures_ready=10/10`
- `generated figure files found=10`
- `figures staged=10`

## Chapter 4 Figures (Regenerated)

Generated under:
- `Prototyping_reformat/thesis_figures/out/figures/`

Staged under:
- `Thesis/markdown/figures/generated/`

Updated files:
- `F02_ch4_core_metric_comparison.png`
- `F03_ch4_radar_profile.png`
- `F04_ch4_remission_slice_comparison.png`
- `F05_ch4_mcnemar_significance_heatmap.png`
- `F06_ch4_confusion_panel.png`

## Supporting Data Exports (New)

- `Prototyping_reformat/thesis_figures/data/ch4_frozen_internal_metrics.csv`
- `Prototyping_reformat/thesis_figures/data/ch4_pass7_drop_subset.csv`
- `Prototyping_reformat/thesis_figures/data/ch4_pass6_mode1_qc.csv`

## Integration Artifacts

- `Prototyping_reformat/thesis_figures/out/freeze_manifest.json`
- `Prototyping_reformat/thesis_figures/out/figure_manifest.csv`
- `Prototyping_reformat/thesis_figures/data/thesis_figure_catalog.csv`
- `Prototyping_reformat/thesis_figures/out/thesis_figure_insertions.md`
- `Prototyping_reformat/thesis_figures/out/part5_thesis_integration_summary.md`

## Notes

- This sync supersedes the Chapter 4 figure legacy warning in Part 3 for `F02`..`F06`.
- Figure IDs remain unchanged for dissertation insertion compatibility.
