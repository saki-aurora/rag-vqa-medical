# Chapter 4 Main Comparison Table

Source: `/home/aristotle/Desktop/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC`. Includes full runs only.

| run_name | model | test_rows | accuracy | macro_f1 | balanced_acc | qwk | mae | rmse | parse_rate |
|---|---|---|---|---|---|---|---|---|---|
| finetune_resnet50 | resnet50_finetune | 1686 | 0.7527 | 0.6800 | 0.6858 | 0.8428 | 0.2533 | 0.5149 | NA |
| finetune_vit_or_swin | vit_or_swin_finetune | 1686 | 0.7129 | 0.6675 | 0.6649 | 0.7642 | 0.3126 | 0.6137 | NA |
| vit_frozen_logreg | vit_frozen_logreg | 1686 | 0.6910 | 0.6192 | 0.6419 | 0.7620 | 0.3458 | 0.6503 | NA |
| resnet50_frozen_logreg | resnet50_frozen_logreg | 1686 | 0.6198 | 0.5346 | 0.5420 | 0.6834 | 0.4324 | 0.7367 | NA |
| vlm_zero_shot_mayo | vlm_zero_shot | 1686 | 0.5486 | 0.1771 | 0.2500 | 0.0000 | 0.6987 | 1.1557 | 1.0000 |
