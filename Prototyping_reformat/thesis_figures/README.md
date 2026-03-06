# Thesis Figures Workspace

This workspace is for thesis figure preparation and tracking.

Part 1 (current scope):
- freeze canonical input artifacts,
- generate input lock files with checksums,
- generate a 10-figure manifest with ready/blocked status.

Part 2 (Chapter 4 figures):
- build Chapter 4 figure-ready tables,
- generate five Chapter 4 figures (`F02` to `F06`),
- write an execution summary report.

Part 3 (Chapter 5 figures):
- build Chapter 5 figure-ready tables,
- generate four Chapter 5 figures (`F07` to `F10`),
- write an execution summary report.

Part 4 (Chapter 3 figure):
- build cross-dataset benchmark tables from frozen Chapter 3 reports,
- generate Chapter 3 figure (`F01`),
- write an execution summary report.

Part 5 (thesis integration):
- stage generated figures into thesis markdown figure directory,
- generate figure catalog with chapter mapping and captions,
- generate ready-to-paste markdown insertion snippets.

## Run Part 1

From repo root:

```bash
python Prototyping_reformat/thesis_figures/freeze_inputs.py
```

```bash
python Prototyping_reformat/thesis_figures/part2_ch4_figures.py
```

```bash
python Prototyping_reformat/thesis_figures/part3_ch5_figures.py
```

```bash
python Prototyping_reformat/thesis_figures/part4_ch3_figures.py
```

```bash
python Prototyping_reformat/thesis_figures/part5_thesis_integration.py
```

## Outputs

- `out/freeze_manifest.json`: machine-readable artifact lock.
- `out/freeze_manifest.csv`: tabular artifact lock.
- `out/figure_manifest.csv`: approved figure plan and readiness.
- `data/ch4_*.csv`: Chapter 4 figure data tables.
- `out/figures/F02*.png` to `out/figures/F06*.png`: Chapter 4 figures.
- `out/part2_ch4_summary.md`: Part 2 output summary.
- `data/ch5_*.csv`: Chapter 5 figure data tables.
- `out/figures/F07*.png` to `out/figures/F10*.png`: Chapter 5 figures.
- `out/part3_ch5_summary.md`: Part 3 output summary.
- `data/ch3_*.csv`: Chapter 3 figure data tables.
- `out/figures/F01*.png`: Chapter 3 figure.
- `out/part4_ch3_summary.md`: Part 4 output summary.
- `data/thesis_figure_catalog.csv`: staged-figure catalog and insertion metadata.
- `out/thesis_figure_insertions.md`: markdown snippets for chapter insertion.
- `out/part5_thesis_integration_summary.md`: Part 5 output summary.

## Notes

- Chapter 4 canonical inputs are pinned to run-level artifacts, including:
  - `vlm_lora_finetune_mayo_balanced_full_20260303`
  - `vlm_zero_shot_mode2_label_sampling_full_20260302`
- Chapter 5 canonical inputs are pinned to `eval_pass4_latest` and pass-4 completion audit.
