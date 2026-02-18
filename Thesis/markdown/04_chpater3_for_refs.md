Chapter 3.	

Investigating Existing VQA Techniques Across GI-Endoscopy Datasets
3.1 Chapter Overview and Evaluation Goal
Chapter 2 surveyed medical VQA technique families and highlighted a central translational gap for GI endoscopy: strong benchmark scores do not automatically imply clinically reliable behavior, especially under class imbalance, domain shift, and high-consequence failure modes [56], [57]. Chapter 3 addresses that gap empirically by auditing persisted experimental artifacts available in this thesis repository and summarizing how model families behave under the dataset constraints that currently exist for colonoscopy-oriented MedVQA.
This chapter is designed as the empirical turning point of the dissertation. The guiding question is not only which model achieves higher aggregate scores, but which model behaviors remain reliable enough to be considered as building blocks for clinician-facing decision support. To answer that question, this chapter evaluates reliability across heterogeneous task regimes that occur in GI MedVQA: (i) closed-label prediction, (ii) open-ended generation, and (iii) ordinal severity assessment.
Chapter 3 overview (reader preview)
Across the datasets in this repository, the chapter provides a reproducible comparison of:
·	constrained or supervised pipelines vs. raw zero-shot generative outputs,
·	dataset- and question-family-specific failure modes (imbalance, format drift, mapping fragility), and
·	severity-oriented reliability under ordinal and clinically meaningful slices (e.g., remission vs. non-remission).
Importantly, Chapter 3 is reproducibility-first: it reports outcomes only where persisted artifacts exist. Where only configuration or status files are present (e.g., scenario YAML definitions), the chapter reports configuration only and does not claim scenario-level performance [72].

Concretely, the objective is to determine, with traceable local evidence, how existing model families behave across:
·	closed-set and binary clinical question answering,
·	open-ended generative answering,
·	severity-oriented ordinal assessment,
·	class-imbalance stress conditions, and
·	scenario-level stress-test protocol definitions for clinically relevant failure probing.
Unlike Chapter 2 (literature synthesis), Chapter 3 is artifact-driven: all results are compiled from saved outputs in the repository, not from newly run training or undocumented reruns [66]-[73]. Where a metric requires predictions that are not persisted, the metric is not reported.
This chapter serves three thesis-level functions:
1.	Establish a reproducible baseline of current GI MedVQA behavior before introducing new methods.
2.	Identify practical reliability bottlenecks (imbalance sensitivity, answer-format instability, lexical/OOV mapping fragility, and clinical threshold failures).
3.	Provide a defensible empirical bridge to Chapter 4, where controlled generative approaches are developed for clinically grounded UC severity rating and later extended toward evidence-aware answering.
Structurally, Section 3.1 defines evidence boundaries and research coverage, Section 3.2 formalizes datasets and model regimes, Section 3.3 specifies evaluation metrics and caveats, Section 3.4 presents dataset-wise results, and Section 3.5 synthesizes cross-dataset findings into thesis-level conclusions.

3.1.1 Evidence Boundary and Reproducibility Scope
Table 3.1 documents the provenance of all primary evidence used in this chapter. For each dataset block, the chapter’s results are compiled from the corresponding report file under the Prototyping_reformat/ tree and the associated persisted outputs referenced by that report [66]-[73].



Dataset / analysis block	Primary report file	Persisted artifact roots	Primary task style
HyperKvasir	Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md	Prototyping_reformat/DatasetAnalysis/HyperKvasir/**/out	23-class visual classification
ImageCLEF MEDVQA-GI 2023	Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md	Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/**/results	closed-label GI VQA
Kvasir-VQA	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/**/out	yes/no and attribute subsets
Kvasir-VQA-x1	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/**/results	large-scale generative + mapped closed-set
LIMUC	Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md	Prototyping_reformat/DatasetAnalysis/LIMUC/**/out	Mayo severity (ordinal 0-3)
Kvasir-SEG (supporting)	Prototyping_reformat/DatasetAnalysis/Kvasir_SEG/Kvasir_SEG.md	Prototyping_reformat/DatasetAnalysis/Kvasir_SEG/0_dataset_prep/**	morphology and mask statistics
Scenario protocol configuration	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/evaluation_comparison/scenarios/scenarios.yaml	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/evaluation_comparison/scenarios/	micro-scenario stress-test definition (configuration only)
Legacy UC generative runtime snapshot (reformatted)	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/evaluation_comparison/phase3_results/summary_uc_phase3.csv	Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/evaluation_comparison/phase3_results/*.csv	open-ended response artifact status
        Table 3.1. Primary Evidence Sources Used in Chapter 3

Reproducibility rules used throughout Chapter 3
·	No undocumented reruns: the chapter reports only what is present in persisted artifacts and their compiled report files.
·	Report-first compilation: each dataset subsection in Section 3.4 is compiled from the dataset’s *.md report file first, then supported by any persisted visualizations or summary outputs referenced by that report.
·	No fabricated comparisons: where a model’s predictions, labels, or scoring outputs are missing, the chapter does not infer or reconstruct results.
·	Configuration-only evidence is treated as configuration: scenario YAML files are reported as protocol definitions, not as evaluated outcomes.
·	Local-report cardinalities: dataset sizes shown in this chapter reflect the counts recorded in the local report artifacts and may differ from official dataset descriptor totals due to subset selection, filtering, or split definitions.
3.1.2 Research Questions Addressed in This Chapter
Chapter 3 provides empirical evidence relevant to the following research questions defined in Chapter 1:
·	RQ2 (comparative reliability): Do constrained/discriminative pipelines remain more reliable than raw zero-shot generative outputs for current GI MedVQA tasks?
·	RQ3 (failure modes): Which failure modes dominate across datasets and task styles (e.g., imbalance collapse, lexical drift, mapping fragility, question-family brittleness)?
·	RQ4 (severity robustness): How reliably can models support UC severity-oriented question answering under ordinal structure and severe-class imbalance?
·	RQ5 (clinical output format) – protocol support only: Chapter 3 includes scenario protocol definitions that inform later clinician-facing evaluation logic, but it does not report scenario outcome metrics unless scored predictions are persisted.
To keep the evidence boundary explicit, scenario-level conclusions are deferred when only scenario configuration is present.

Chapter 2 established the technique landscape (discriminative fusion models, transformer-based multimodal encoders, generative multimodal models, and retrieval-augmented directions) and emphasized that evaluation must be aligned with clinical risk, including lessons from early medical VQA challenge settings [57], [58], [59]. Chapter 3 operationalizes that principle by evaluating model families using the datasets and artifact outputs available in this repository. The chapter therefore functions as the thesis’s empirical “reality check”: it identifies what is currently reliable under GI-specific constraints and what failure patterns must be addressed before introducing controlled generative and evidence-aware extensions in subsequent chapters.

3.2 Experimental Scenarios and Data Regimes
Section 3.2 defines the empirical scope: which datasets are evaluated, what task styles they represent, and which model families are compared. The aim is not to claim universal generalization across all GI settings, but to characterize reliability patterns within the available artifact evidence and across diverse task regimes (closed-label, generative, ordinal).
3.2.1 Dataset-Task Matrix
Table 3.2 summarizes the datasets included in Chapter 3 and the primary role each dataset plays in the evaluation. The “cardinality” values reflect the counts recorded in the local report files used for this chapter [66]-[70].
Dataset	Cardinality in local report	Core outputs	Main evaluation axis	Clinical relevance
HyperKvasir	10,662 images, 23 classes	class labels	multiclass reliability under imbalance	broad GI visual grounding
ImageCLEF MEDVQA-GI 2023	36,683 QA rows (29,351 train, 7,332 val)	label IDs per question	per-question closed-label VQA reliability	benchmarked GI QA consistency
Kvasir-VQA	58,849 QA rows, 6,500 images	yes/no, attributes, free text	subset reliability and format stability	colonoscopy QA behavior
Kvasir-VQA-x1	159,549 QA rows, 6,449 images	free-text answers + mapped labels	generative fidelity, complexity effects	robust MedVQA reasoning stress
LIMUC	11,276 images, Mayo 0-3	ordinal severity class	macro-F1, QWK, remission slice	UC treatment-aligned severity
Kvasir-SEG	1,000 image-mask pairs	mask morphology stats	coverage/shape support metrics	future localization grounding
Table 3.2. Dataset and Task Matrix for Chapter 3 Experiments
Why these datasets are included
·	Kvasir-VQA provides the most direct colonoscopy-oriented VQA evidence and is used to examine subset reliability and answer-format stability under common question families [52], [68].
·	ImageCLEF MEDVQA-GI 2023 provides a standardized closed-label GI VQA setting where fine-tuned VQA models can be compared to zero-shot generative outputs under label constraints [54], [67].
·	Kvasir-VQA-x1 provides a large-scale generative stress test, useful for studying overlap metrics, complexity effects, and label mapping fragility in free-text answers [53], [69].

·	LIMUC anchors the thesis’s clinical flagship: UC severity, where ordinal structure and clinically meaningful slices (e.g., remission vs non-remission) are essential for interpretation [55], [70].
·	HyperKvasir provides broad GI visual grounding with strong long-tail imbalance, allowing analysis of head–tail recall asymmetry that can be hidden by aggregate accuracy [51], [66].
·	Kvasir-SEG is included as supporting evidence for morphology/localization-oriented statistics that later motivate grounding-aware directions; it is not treated as a primary VQA benchmark in this chapter [71].
3.2.2 Model Families Compared
The model comparisons in Chapter 3 reflect what is available as persisted artifacts rather than an exhaustive survey of all possible architectures. The goal is to compare representative families that span the major design choices relevant to clinical reliability: supervised classification, constrained VQA, and open-ended generation.
Family	Example persisted models	Typical answer mode
Supervised CNN/ViT classifiers	resnet50_supervised, vit_supervised, finetune_resnet50	closed label
Frozen encoder + shallow classifier	vit_frozen_logreg, clip_linear_baseline, resnet50_frozen_logreg	closed label
Classical multimodal fusion	resnet_gru_m1_*, vit_bertlite_m2_*	closed set
Transformer VQA fine-tuned	vilt_finetune	closed label per question
Zero-shot VLM/MLLM	qwen2_5_vl_zeroshot, medgemma_zeroshot, blip2_zero_shot	free text (optionally projected)
Parameter-efficient adaptation	medgemma_lora_original, qwen2_5_vl_lora_finetune (logs persisted)	free text
Table 3.3. Representative Model Families in Persisted Artifacts
Interpretation guide
·	Closed-label families (supervised or shallow classifiers) provide a reliability baseline where output space is controlled and evaluation is deterministic.
·	Classical fusion and transformer VQA models represent structured MedVQA pipelines designed to combine image and question representations under a constrained output space.
·	Zero-shot VLM/MLLM models represent open-ended generation, which may be more flexible but can be unstable in output format and lexical grounding.
·	Parameter-efficient adaptation captures whether lightweight fine-tuning improves generative fidelity without requiring full retraining.
Section 3.3 defines the metric bundles used to compare these families and clarifies how raw vs projected scoring is treated when free text must be mapped to a constrained answer space.
3.2.3 Data Profile Figures (Kvasir-VQA)
Before presenting results, the chapter includes Kvasir-VQA profiling figures because they explain several behaviors observed later in Section 3.4, particularly the dominance of yes/no questions and the presence of template skew. These distributional properties can inflate superficial performance on majority question types while masking failure on rarer but clinically important categories [68], [74]-[76].

Figure 3.1 QA count by source domain 
Figure 3.1 summarizes how QA pairs are distributed across source domains in the Kvasir-VQA preparation artifacts [74]. Source-domain imbalance implies that model performance may reflect dominant sources disproportionately, and it motivates later per-family and per-task reporting rather than single aggregate scores.

Figure 3.2 QA count by type distribution 
The distribution shows that yes/no questions form a large share of Kvasir-VQA [75]. This creates a majority-family regime where models can perform strongly by exploiting priors, and it motivates the use of balanced metrics and subset-level reporting to avoid overstating reliability.

Figure 3.3 Question type by source
Question-type composition varies by source domain, indicating that the dataset is not homogeneous across origins [74]. This supports the chapter’s emphasis on failure-mode analysis and warns against interpreting a single “overall” score as uniformly applicable across question families.

Figure 3.4 Answer type distribution
The answer-type distribution highlights why evaluation must be matched to task style: exact-match metrics are natural for constrained outputs but can be overly harsh for free-text generation, where token-level overlap diagnostics may be more informative [76]. This motivates the layered metric design defined in Section 3.3.
Signal	Value
Total QA rows	58,849
Unique images	6,500
Mean QA rows per image	9.05
Yes/No questions	26,515 (45.06%)
Entity questions	10,528 (17.89%)
Counting questions	10,118 (17.19%)
Location questions	8,424 (14.31%)
Table 3.4 Kvasir-VQA Distribution Snapshot
The strong question-family skew supports two design choices used throughout Chapter 3: (i) report class-balanced or family-aware metrics where possible, and (ii) interpret aggregate scores cautiously, because strong performance on a dominant family (e.g., yes/no) does not guarantee reliable behavior on count, location, or clinically higher-risk categories [56].
3.2.4 Scenario Micro-Benchmark Definition
To connect benchmark evaluation to realistic clinical vignettes, the repository includes a micro-scenario protocol definition [72].
The protocol defines three focused scenario templates:
1.	S1: active bleeding binary detection,
2.	S2: instrument type plus polyp count,
3.	S3: Paris morphology closed set.
These scenarios are intentionally small and are treated as stress vignettes, not as statistical benchmark replacements. Only scenario configuration is persisted in the reformatted evidence tree; scored scenario predictions are not persisted [72]. Therefore, Chapter 3 reports scenario definitions as protocol context only and does not report scenario outcome metrics in the absence of scored artifacts.
3.2.5 Alignment with Prior Dissertation Problem Settings
The dataset and task choices in Chapter 3 are designed to align with common GI-endoscopy AI concerns while focusing on MedVQA-specific reliability:
·	Severity and treatment relevance are represented by LIMUC, where ordinal behavior and clinically interpretable slices are necessary for credible UC severity assessment [55], [62]-[65], [70].
·	Robustness under heterogeneity is addressed by including multiple datasets with different answer spaces and task formulations, enabling cross-dataset comparison of failure patterns rather than optimizing for a single benchmark.
·	Transparency of model behavior is operationalized by reporting not only aggregate metrics but also imbalance-aware indicators, question-family slices, and explicit boundaries where evidence is unavailable.
This chapter does not attempt to replace established GI work on detection, segmentation, or video-level robustness. Instead, it complements that literature by adding a dedicated empirical layer for question-answer reliability, which is a prerequisite for interactive, clinician-facing MedVQA systems.
3.3 Evaluation Metrics and Statistical Protocol
This chapter compares model families across heterogeneous GI MedVQA task regimes. Because these regimes differ in answer format (fixed labels vs free text) and label structure (nominal vs ordinal), no single metric adequately captures reliability. Chapter 3 therefore uses a layered metric bundle and a conservative reporting protocol: point estimates are reported only when corresponding persisted predictions and evaluation outputs exist, and comparative claims are bounded to the evidence available in the persisted artifacts [56].

   Figure 3.5 — Artifact-driven benchmarking workflow
3.3.1 Metric Layers
To avoid single-metric bias, Chapter 3 uses four metric layers depending on task style and clinical risk profile.
(A) Closed-set classification and constrained QA 
These metrics are used when outputs are categorical and the answer space is controlled (e.g., HyperKvasir class labels, ImageCLEF label IDs, constrained Kvasir-VQA subsets).
·	Accuracy: overall fraction of correct predictions.
·	Macro-F1: unweighted mean of per-class F1, emphasizing minority-class behavior.
·	Balanced accuracy: mean of per-class recall (equally weights classes regardless of support).
·	MCC (Matthews correlation coefficient): correlation-style score robust under imbalance (reported where available in persisted outputs).
·	Cohen’s kappa: agreement measure corrected for chance (reported where available in persisted outputs).
·	Imbalance diagnostics (where included in artifacts): head–tail or rare–common recall gaps computed from classwise recall slices (e.g., thresholds defined in the dataset’s local report).
(B) Generative overlap and format fidelity (free-text outputs)
These are used when models generate natural language answers rather than selecting from a fixed ontology. In this chapter they are treated as diagnostic fidelity indicators, not as sufficient evidence of clinical correctness [57], [60], [61].
·	Exact match (EM): strict string-level match under the normalization used by the local evaluation scripts.

·	Token-F1: token-level overlap between generated and reference answers, useful when exact match is too strict.
·	ANLS: edit-distance-based similarity score used in VQA-style evaluation; implementation details follow the persisted evaluation outputs and scripts used to produce the repository artifacts.
·	BLEU / ROUGE-L / METEOR (where available): standard NLG overlap metrics reported only when computed in the persisted artifacts.
(C) Ordinal severity and clinically meaningful slices (UC severity as flagship use case)
These metrics are used when labels have an ordinal relationship (e.g., Mayo 0–3) and when the clinical decision boundary matters more than overall accuracy.
·	QWK (quadratic weighted kappa): penalizes ordinal disagreements proportionally to distance (e.g., confusing 0↔1 is not equivalent to 0↔3).
·	MAE / RMSE: absolute and squared error over ordinal labels treated as numeric.
·	Spearman correlation (where available): rank correlation for ordinal consistency.
·	Clinical remission slice: a thresholded binary evaluation derived from ordinal severity labels (in LIMUC artifacts, remission is evaluated as Mayo 0–1 vs 2–3, as indicated in the corresponding table caption).
(D) Uncertainty and comparative significance diagnostics (paired predictions only)
To avoid overinterpreting small point-estimate differences, Chapter 3 uses statistical diagnostics only when paired predictions are persisted for the same evaluation set.
·	Wilson confidence intervals: used for proportions (e.g., accuracy) when included in the report outputs.
·	Paired McNemar test: used for paired comparison of two classifiers on the same examples; reported only where persisted counts (n01, n10) and p-values exist in the artifacts.



Scenario type	Primary metrics	Why these metrics
Binary clinical detection	recall/sensitivity, macro-F1, MCC	false negatives and imbalance sensitivity
Multiclass closed-set QA	accuracy + macro-F1 + balanced accuracy	aggregate plus per-class fairness
Free-text QA	token-F1 + ANLS + overlap metrics	lexical similarity with tolerance to paraphrase
Ordinal severity	QWK + MAE/RMSE + remission slice	ordinal penalty and clinical thresholding
Model comparison claims	McNemar + CIs	avoids overinterpreting point estimates
Table 3.5 Metric Selection by Scenario Type
Throughout Chapter 3, N denotes the number of evaluated items recorded in the local report (images or QA rows). Metrics are interpreted primarily within-dataset because answer spaces and task definitions differ across datasets.
3.3.2 Important Evaluation Caveats
This chapter’s evaluation design is intentionally conservative. The following caveats define how results should be interpreted and what is not claimed.
1.	Cross-dataset scores are not directly comparable. Different datasets implement different answer spaces, label ontologies, and question-family distributions. Therefore, Chapter 3 uses cross-dataset comparison mainly to identify consistent patterns of failure modes, not to rank models globally.
2.	Generative overlap is not clinical correctness. Metrics such as token-F1, ANLS, BLEU, ROUGE-L, and METEOR quantify lexical similarity and answer-format adherence. They do not guarantee that generated text is clinically faithful or visually grounded [57], [60], [61]. In this chapter, generative overlap metrics are treated as diagnostic indicators and interpreted alongside failure signals (unknown/OOV behavior, mapping fragility, and question-family brittleness).
3.	Raw vs. projected scoring is explicitly separated (diagnostic only). Some persisted artifacts include both “raw” scoring of generated outputs and a secondary “projected” diagnostic that maps free text into a constrained answer space before scoring. This chapter follows the artifact semantics:
o	Raw scoring reflects direct evaluation against canonical targets under the local evaluation convention; for many label-ID tasks, raw free-text outputs can collapse to near-zero label accuracy because the output is not in the canonical label format.
o	Projected scoring deterministically maps generated text to the allowed answer set (e.g., canonical labels for a question family) using the repository’s evaluation logic, then scores the mapped label.
o	Unknown/OOV handling: if an output cannot be mapped to the allowed answer set, it is treated as unknown/OOV for that diagnostic and contributes to unknown/OOV rates where reported.
Interpretation rule: projected scores are used to quantify format and ontology alignment under deterministic mapping; they are not treated as a replacement for primary reliability evidence.
1.	Scenario results are not claimed without persisted scenario predictions. Scenario YAML definitions are included as protocol context, but scenario outcome metrics are reported only if scored predictions are persisted. If only configuration is present, Chapter 3 reports configuration only and defers scenario outcomes.
2.	Statistical tests are reported only where paired predictions exist. McNemar tests and confidence intervals require paired predictions for the same examples. Where the persisted artifacts include these counts and p-values, Chapter 3 reports them. Where they are not present, Chapter 3 reports point estimates without significance claims.
3.	Missing metrics are treated as “not computed,” not “zero.” If a metric is not applicable or not computed in the persisted evaluation outputs (e.g., BLEU/ROUGE absent for some runs), the chapter reports it as unavailable and does not infer values.

Figure 3.6 — Label projection / answer normalization pipeline (raw vs projected + unknown/OOV)
3.3.3 Metric Design Choices Grounded in Prior Work
The metric stack in this chapter follows a common principle in clinical AI evaluation: performance should be characterized using both aggregate discrimination and robustness under imbalance and clinically relevant thresholds, rather than relying on a single headline score [56].
For GI endoscopy tasks, this implies:
·	macro-F1 and balanced accuracy are emphasized to prevent majority-class dominance from masking clinically relevant failures;
·	MCC and kappa are reported where available because they capture agreement behavior under imbalance;
·	ordinal metrics and clinical slices (QWK, MAE/RMSE, and remission-style thresholds) are required for UC severity tasks where the distance between grades and the decision boundary are clinically meaningful.

For generative tracks, overlap metrics are retained as format and lexical-fidelity diagnostics, but they are not treated as sufficient endpoints for clinical reliability. Where projection/mapping diagnostics are available, they are reported explicitly as diagnostics to quantify ontology alignment and unknown/OOV behavior.
Finally, comparative statistics (e.g., McNemar) are used as safeguards against overinterpreting small differences, but only where paired evidence is persisted. The chapter therefore prioritizes traceable, artifact-backed reporting over completeness of metric coverage.
3.4 Baseline and Existing-Model Results
Section 3.4 reports dataset-wise results compiled from persisted artifacts described in Table 3.1 [66]-[73]. Because each dataset defines different answer spaces (class labels, label IDs per question, or free-text answers) and different task regimes (classification, closed-label VQA, subset QA), results are interpreted primarily within each dataset. Cross-dataset comparisons are used to identify consistent reliability patterns (e.g., imbalance sensitivity, answer-format instability), rather than to produce a single global ranking.
Each dataset subsection follows a consistent structure: (i) role of the dataset in the thesis evaluation design, (ii) which model families are compared and why, (iii) interpretation of the reported tables (overall metrics, robustness slices, and where available paired diagnostics), and (iv) limitations and implications for subsequent chapters.
3.4.1 HyperKvasir: 23-Class GI Image Classification
Although HyperKvasir is not a VQA dataset, it provides an important visual grounding stress test for GI imaging [51], [66]. The task here is 23-class GI image classification, which functions as a proxy for how well different visual backbones and representations capture GI categories under realistic data constraints. The key difficulty is strong long-tail imbalance: in the persisted report, test-set class support ranges from 1 to 115 per class (≈115× ratio), meaning that a model can achieve high aggregate accuracy while still failing on rare classes that may be clinically important.
Table 3.6 summarizes overall test performance for supervised and frozen-feature baselines, alongside a zero-shot VLM baseline projected into a label space (as recorded in the persisted artifact outputs).

Model	Accuracy	Balanced Acc	Macro-F1	MCC	Kappa
resnet50_supervised	0.8789	0.6266	0.5943	NA	NA
vit_supervised	0.8714	0.5391	0.5242	NA	NA
vit_frozen_logreg	0.8620	0.6130	0.6052	0.8505	0.8504
clip_linear	0.8620	0.5799	0.5721	0.8503	0.8503
blip2_zero_shot_clip	0.0638	0.0529	0.0254	0.0386	0.0303
Table 3.6 HyperKvasir Overall Test Metrics
Table 3.6 summarizes overall test performance for supervised and frozen-feature baselines, alongside a zero-shot VLM baseline projected into a label space (as recorded in the persisted artifact outputs).
The supervised and frozen-feature pipelines form a strong reliability baseline on aggregate metrics, while the zero-shot VLM baseline performs poorly on this closed-label classification setting. However, aggregate metrics alone are insufficient in a long-tail regime. In particular, balanced accuracy and macro-F1 are emphasized here because they weight minority classes more fairly than raw accuracy.
To quantify long-tail robustness explicitly, Table 3.7 reports rare-class and common-class recall slices (as defined in the persisted report), along with a head–tail recall gap.
Model	Rare-class recall (support <= 5)	Common-class recall (support >= 90)	Common-minus-rare gap
resnet50_supervised	0.1595	0.9375	0.7779
vit_supervised	0.0000	0.9432	0.9432
vit_frozen_logreg	0.1714	0.9120	0.7406
clip_linear	0.0476	0.9091	0.8615
blip2_zero_shot_clip	0.0000	0.0471	0.0471
Table 3.7 HyperKvasir Imbalance Robustness Slices
All high-accuracy models exhibit substantial head–tail asymmetry: common-class recall is high while rare-class recall remains low. Notably, some models achieve strong common-class recall while still scoring near zero on rare-class recall, illustrating why long-tail evaluation slices are required for risk-aware interpretation. This result motivates later chapters to treat minority-class behavior as a first-class reliability requirement, rather than assuming that high overall accuracy implies safe performance.
Where paired predictions are available in the artifacts, Table 3.8 reports selected McNemar tests to assess whether differences between model variants are likely to be noise-level fluctuations on the same test examples.


Pair	n01 (A wrong, B right)	n10 (A right, B wrong)	p-value
vit_frozen_logreg vs clip_linear	68	68	0.931666
vit_frozen_logreg vs blip2_zero_shot_clip	21	871	< 1e-6
clip_linear vs blip2_zero_shot_clip	19	869	< 1e-6
Table 3.8 HyperKvasir Pairwise McNemar Tests
In the persisted paired comparisons, vit_frozen_logreg and clip_linear show no meaningful difference (symmetric disagreement counts and high p-value), while both are strongly separated from the zero-shot baseline (highly asymmetric disagreement counts and very small p-values). These tests support the qualitative conclusion already visible in Table 3.6: within this dataset and evidence scope, supervised/frozen discriminative pipelines dominate the zero-shot baseline.
Limitations and implications. HyperKvasir is used here as a GI visual grounding stress test rather than a direct VQA benchmark. Therefore, its primary value is diagnostic: it demonstrates that (i) strong aggregate performance can coexist with poor rare-class recall, and (ii) reliable GI perception under long-tail imbalance remains a bottleneck that must be monitored explicitly. This motivates later chapters to preserve strong visual grounding components and to treat class-imbalance mitigation as necessary for clinically aligned behavior.
3.4.2 ImageCLEF MEDVQA-GI 2023: Closed-Label GI VQA
ImageCLEF MEDVQA-GI 2023 provides a closed-label VQA setting in which each question is associated with a constrained answer ID space [54], [67]. This enables deterministic evaluation and is well-suited for testing the thesis’s core reliability question: how do fine-tuned multimodal VQA models compare to raw zero-shot generative VLM outputs under label constraints? In this chapter, evaluation is reported for the persisted validation split (N = 7,332) recorded in the local report artifacts.
Table 3.9 reports overall validation metrics for a fine-tuned transformer VQA model and two zero-shot variants: “raw” (direct outputs) and “projected” (diagnostic mapping of outputs into the allowed answer space, as defined in Section 3.3).

Model variant	N	Accuracy	Balanced Acc	Macro-F1	MCC	Kappa
vilt_finetune	7,332	0.9089	0.5853	0.5823	0.8876	0.8875
qwen2_5_vl_zeroshot_raw	7,332	0.0007	0.0433	0.0007	-0.0696	-0.0626
qwen2_5_vl_zeroshot_projected	7,332	0.0670	0.0899	0.0379	-0.0296	-0.0278
Table 3.9 ImageCLEF MEDVQA-GI 2023 Validation Metrics
The fine-tuned VQA model exhibits strong closed-label accuracy, while the raw zero-shot generative outputs score near zero under the label-ID evaluation format. Projection improves the zero-shot diagnostic score, indicating that some generated outputs can be mapped into the label space, but the projected accuracy and macro-F1 remain far below the fine-tuned baseline. The negative MCC/kappa values for the zero-shot variants are consistent with systematic mismatch under the constrained label-ID evaluation format, rather than random variation.
To avoid hiding question-family brittleness, Table 3.10 reports family-level aggregates. This is important because some families (e.g., procedure or attribute) may require stricter ontology adherence than others.
Question family	Rows	ViLT acc	Qwen raw acc	Qwen projected acc	ViLT macro-F1	Qwen projected macro-F1
attribute	1,600	0.8488	0.0000	0.0225	0.5153	0.0340
binary/boolean	2,800	0.9339	0.0004	0.1168	0.9222	0.0906
count	1,200	0.9008	0.0017	0.0400	0.3950	0.0158
location	1,332	0.9092	0.0015	0.0601	0.3115	0.0205
procedure	400	0.9975	0.0000	0.0000	0.9969	0.0000
Table 3.10 Family-Level Aggregates on ImageCLEF Validation
Two patterns are visible. First, the fine-tuned model remains consistently strong across families in accuracy terms, but macro-F1 varies substantially across families (notably count/location), indicating that even strong aggregate accuracy can conceal intra-family imbalance or difficulty. Second, the zero-shot model’s raw outputs perform near zero across families, and projection yields partial recovery mainly for binary/boolean questions, with minimal gains for attribute, count, and location and no recovery for the procedure family in the persisted artifacts. This indicates that projection can rescue some format mismatch, but it does not resolve the core reliability gap under structured families.
To illustrate where projection helps most, Table 3.11 lists the questions with the largest raw→projected accuracy gains in the persisted validation outputs.


Question	Raw acc	Projected acc	Absolute gain
Is there a green/black box artefact?	0.0000	0.5475	+0.5475
Are there any instruments in the image?	0.0000	0.1800	+0.1800
Where in the image is the abnormality?	0.0000	0.1400	+0.1400
What color is the abnormality?	0.0000	0.0800	+0.0800
How many polyps are in the image?	0.0025	0.0600	+0.0575
         Table 3.11. Largest Qwen Lexical-Projection Gains (Validation Accuracy)
These examples show that deterministic mapping can substantially improve measurable label accuracy for specific prompts, suggesting that the raw outputs sometimes contain recoverable information that is not expressed in the expected label form. However, as emphasized in Section 3.3, projection remains a diagnostic and can overestimate correctness when matches are shallow or purely lexical. The key reliability result for this chapter is therefore the persistent gap in Table 3.9 and Table 3.10: even after projection, zero-shot outputs remain far below a fine-tuned closed-label baseline across major question families.
Limitations and implications. This subsection compares a fine-tuned transformer baseline to one zero-shot model under the persisted evaluation configuration. It does not claim universal generalization to all VLMs or all GI datasets. Moreover, because paired statistical tests are not reported for this dataset in the included tables, this chapter avoids “significance” wording here and interprets differences as large observed gaps under the available evidence. The practical implication is that for label-ID GI VQA settings, reliable performance requires answer-space governance (constrained decoding, ontology compliance) and likely supervision or adaptation; these conclusions motivate the controlled generative design decisions introduced in Chapter 4.
3.4.3 Kvasir-VQA: Subset Reliability and Answer-Format Stability
Kvasir-VQA is a colonoscopy-oriented VQA resource and is central to the supervisor-directed empirical pathway for this dissertation [52], [68]. The persisted artifacts available for this chapter include (i) structured subset evaluations for yes/no and attribute-style questions and (ii) a reformatted phase-3 generative snapshot intended to capture runtime status [73]. This subsection focuses on two reliability themes relevant to GI MedVQA: (1) how strong constrained/fusion pipelines can be on structured question families, and (2) how fragile unconstrained generation can become when output format is not governed.

Model	N	Accuracy	Balanced Acc	Macro-F1	MCC	Unknown rate
resnet_gru_m1_yesno	443	0.986456	0.973673	0.964953	0.930163	0.0000
vit_bertlite_m2_yesno	443	0.950339	0.906593	0.878126	0.759432	0.0000
blip2_zeroshot_yesno	443	0.893905	0.509376	0.492332	0.086138	0.0000
blip_vqa_base_yesno_forced_choice	12,267	0.518301	0.514587	0.502888	0.030894	0.0000
blip_vqa_base_yesno_freegen	500	0.000000	NA	0.000000	NA	1.0000
            Table 3.12 Kvasir-VQA Yes/No Results
Two distinct patterns appear. First, constrained fusion models (resnet_gru_m1_yesno, vit_bertlite_m2_yesno) achieve high accuracy and high balanced metrics on this persisted subset, indicating that structured yes/no questions can be handled reliably when the output space is controlled. Second, the zero-shot model (blip2_zeroshot_yesno) achieves high accuracy but near-chance balanced accuracy and low MCC, which is consistent with majority-class bias in an imbalanced yes/no distribution—an example of why macro-F1, balanced accuracy, and MCC are necessary complements to raw accuracy. Finally, the free-generation variant (blip_vqa_base_yesno_freegen) collapses under this evaluation format with unknown-rate = 1.0, indicating complete answer-format noncompliance under the persisted scoring rules.
Table 3.13 reports results for an attribute subset, which is more challenging than yes/no because it typically expands the answer vocabulary and increases ambiguity.

Model	N	Accuracy	Balanced Acc	Macro-F1
resnet_gru_m1_attribute	352	0.670455	0.376936	0.367341
vit_bertlite_m2_attribute	352	0.656250	0.374696	0.355586
            Table 3.13 Kvasir-VQA Attribute Subset
While accuracy remains moderate, balanced accuracy and macro-F1 are substantially lower, indicating that the attribute subset exhibits stronger imbalance and/or class confusability than the yes/no subset. This aligns with the broader reliability view adopted in this dissertation: even when constrained pipelines perform strongly on a dominant family (yes/no), harder families can remain bottlenecks and require separate analysis rather than extrapolation from the easiest subset.
This subsection reports subset-level evidence rather than a full Kvasir-VQA benchmark sweep. The yes/no subset (N=443) and the larger forced-choice evaluation (N=12,267) are not directly comparable as a controlled experiment because they differ in evaluation set size and possibly in subset construction. Nevertheless, the reliability message is consistent across the persisted artifacts: (i) constrained/fusion pipelines can be very strong on structured subsets, and (ii) unconstrained generation can collapse under closed-label evaluation due to answer-format instability. These results motivate the design stance used in subsequent chapters: preserve constrained pathways for high-reliability sub-questions and introduce generative answering only under explicit output governance (constraints, mapping, or abstention behavior).
3.4.4 Kvasir-VQA-x1: Large-Scale Generative Reasoning Benchmark
Kvasir-VQA-x1 is the largest QA setting used in this thesis repository evidence and serves as the strongest stress test for open-ended generative answering under increased question variety and difficulty [53], [69]. Unlike closed-label tasks (where answers are selected from a fixed ontology), Kvasir-VQA-x1 evaluates free-text generation. As a result, reliability in this setting depends not only on underlying visual reasoning, but also on answer-format stability, lexical normalization, and ontology alignment—all of which can fail even when a model produces plausible text.
This subsection reports three complementary views of behavior:
1.	Generative overlap diagnostics quantify lexical similarity between generated text and references under the evaluation conventions used in the persisted artifacts.
2.	Closed-set style diagnostics and baselines illustrate how mapping and answer-space governance can affect apparent performance, including failure patterns such as high OOV (out-of-vocabulary) or unknown mapping rates.
3.	Complexity-level breakdown provides a descriptive diagnostic of how token-level overlap varies with question complexity (as recorded in the persisted summaries).
Generative overlap and adaptation effects
Model	EM	Token-F1	ANLS	BLEU	ROUGE-L	METEOR	Count
medgemma_lora_original	0.000000	0.508473	0.340755	NA	NA	NA	15,955
medgemma_zeroshot	0.000063	0.213080	0.017498	0.033341	0.158501	0.141180	15,955
llava_zeroshot	0.000000	0.212437	0.007032	0.025760	0.163942	0.150097	15,955
qwen2_5_vl_zeroshot	0.000000	0.172788	0.000000	0.017084	0.123496	0.187288	15,955
            Table 3.14 Kvasir-VQA-x1 Generative Metrics
Exact match (EM) is near zero across all runs, which is consistent with strict string matching under free-text generation where minor lexical differences can prevent EM even when the meaning is similar. Token-F1 and ANLS provide more tolerant diagnostics and show a clear adaptation effect: the LoRA-adapted run substantially improves token-level overlap compared with the zero-shot runs in the persisted artifacts. Where BLEU/ROUGE/METEOR are marked NA, the metric was not computed in the persisted outputs and is not inferred.
Reliability boundary. These overlap metrics quantify surface-level fidelity and answer-format adherence, but they do not guarantee clinical correctness or visual grounding (Section 3.3) [57], [60], [61]. Accordingly, they are treated as diagnostic indicators and interpreted alongside mapping/OOV signals and class-balanced behavior.
Closed-set style baselines and mapping fragility
To expose how answer-space governance affects apparent performance, the persisted artifacts include additional diagnostics and baselines that map outputs into constrained spaces or evaluate alternative non-generative approaches.
Model	N	Accuracy	Balanced Acc	Macro-F1	Notes
fusion_tfidf_vit_logreg	5,893	0.814865	NA	0.150749	fusion baseline
text_yesno_tfidf_logreg	1,540	0.777922	NA	0.777291	yes/no-specific
vlm_zeroshot_label_mapped	15,955	0.561642	NA	0.005817	OOV rate 0.973
text_topk_tfidf_logreg	4,252	0.422389	0.235698	0.204103	top-3 0.7408
image_resnet50_logreg	5,952	0.233535	NA	0.008556	image-only
text_bert_classifier	9,148	0.158942	0.006654	0.002327	weak generalization
image_vit_logreg	4,252	0.020461	NA	0.006352	image-only
           Table 3.15 Kvasir-VQA-x1 Mapped/Baseline Diagnostics


Two reliability signals are important here:

1.	Moderate mapped accuracy can be misleading. The mapped VLM diagnostic reports accuracy above 0.5 while macro-F1 is near zero, alongside an extremely high OOV rate. This indicates severe answer-space mismatch: many outputs cannot be mapped to the expected label space, and the mapped subset does not support balanced performance. As emphasized in Section 3.3, mapped accuracy is a format/ontology diagnostic, not evidence of clinically reliable reasoning.
2.	Subset-specific baselines can look strong due to question-family skew. The yes/no-specific baseline reports high macro-F1 within that restricted subset, reinforcing the need to report results by task regime and avoid extrapolating from easier subsets to overall reliability.
Complexity effects (descriptive diagnostic)
Complexity	llava_zeroshot	medgemma_zeroshot	qwen2_5_vl_zeroshot
1	0.151298	0.145365	0.079875
2	0.217874	0.216663	0.171684
3	0.271474	0.280927	0.271952
            Table 3.16 Token-F1 by Complexity Level
Token-F1 increases with complexity level in these persisted summaries. This trend is reported descriptively only: complexity can correlate with longer outputs or repeated lexical cues that inflate overlap scores without necessarily indicating better grounded reasoning. Therefore, complexity-level overlap trends should be interpreted alongside OOV behavior and family-level diagnostics.
Limitations and implications. Kvasir-VQA-x1 demonstrates a common generative MedVQA pattern under repository evidence: strict EM collapses; token-level overlap improves with adaptation; and deterministic mapping can yield moderate accuracy while macro-F1 remains extremely low due to OOV behavior and answer-space skew. The practical implication for Chapter 4 is that generative capability must be introduced with answer-space governance (controlled decoding, normalization/mapping safeguards, and abstention behavior) rather than being used as an unconstrained answer path.
3.4.5 LIMUC: UC Severity Reliability (Flagship Clinical Axis)
LIMUC provides the most clinically direct evidence in Chapter 3 because it targets UC severity grading (Mayo 0–3), which is ordinal and clinically high-stakes [55], [62]-[65], [70]. In this setting, reliability cannot be summarized by accuracy alone: ordinal distance matters (0↔1 differs from 0↔3), and clinically meaningful threshold behavior (e.g., remission vs active disease) must be audited explicitly. Therefore, this subsection emphasizes ordinal agreement (QWK), error magnitude (MAE/RMSE), and a remission-oriented slice alongside standard classification metrics.
Ordinal and classification metrics
Model	Accuracy	Balanced Acc	Macro-F1	QWK	MAE	RMSE
finetune_resnet50	0.753855	0.695008	0.682889	0.835097	0.256821	0.528545
finetune_vit_or_swin	0.727165	0.673848	0.672142	0.806259	0.287070	0.564888
vit_frozen_logreg	0.689798	0.641650	0.618454	0.758806	0.348161	0.654848
clip_linear_baseline	0.679122	0.635709	0.602016	0.745502	0.367734	0.687112
resnet50_frozen_logreg	0.619217	0.542258	0.533958	0.679280	0.434757	0.742299
vlm_zero_shot_mayo	0.548636	0.250000	0.177135	0.000000	0.698695	1.155727
            Table 3.17 LIMUC Overall Test Metrics
Fine-tuned visual backbones lead across accuracy, macro-F1, ordinal agreement (QWK), and error magnitude (MAE/RMSE), indicating more consistent severity grading and fewer large ordinal errors. The zero-shot severity prompt baseline shows substantially weaker balanced metrics and collapses on ordinal agreement (QWK = 0.0) under the persisted evaluation configuration, indicating that ordinal consistency is not preserved under this zero-shot prompting setup.
Clinical threshold slice (remission vs active disease)
Because clinical workflows often use thresholded reasoning (e.g., remission vs active disease), the persisted artifacts include a remission-oriented slice (as defined in the LIMUC report).
Model	Remission accuracy	Sensitivity	Specificity	Remission F1
finetune_resnet50	0.947805	0.967603	0.855219	0.968300
finetune_vit_or_swin	0.937722	0.968323	0.794613	0.962433
vit_frozen_logreg	0.902135	0.917207	0.831650	0.939182
resnet50_frozen_logreg	0.886714	0.917927	0.740741	0.930317
clip_linear_baseline	0.886714	0.895608	0.845118	0.928705
vlm_zero_shot_mayo	0.823843	1.000000	0.000000	0.903415
            Table 3.18 LIMUC Remission Slice Metrics
Tuned and frozen-feature baselines maintain both sensitivity and specificity at usable levels under this slice. In contrast, the zero-shot baseline exhibits a critical threshold failure: sensitivity is 1.0 while specificity is 0.0, indicating complete inability to discriminate the negative class under this operational definition. This is precisely the kind of high-risk behavior that accuracy alone can hide and illustrates why clinical slice analysis is required for severity-oriented MedVQA.

              Figure 3.7 — UC severity evaluation workflow (LIMUC)
Per-class bottlenecks
To make minority-class behavior explicit, the persisted artifacts include a best-per-class summary.
Mayo class	Support	Best model	Best F1	Best recall
0	925	finetune_resnet50	0.852516	0.796757
1	464	finetune_resnet50	0.683859	0.771552
2	177	finetune_resnet50	0.552326	0.536723
3	120	finetune_vit_or_swin	0.683544	0.675000
             Table 3.19 LIMUC Per-Class Best Model Summary
Support is strongly imbalanced across Mayo classes, and performance remains weakest for intermediate-to-severe classes (notably Mayo 2) even for the best-performing models. This supports the thesis motivation that minority severe classes remain the principal reliability bottleneck and require explicit attention in both evaluation and method design.
Limitations and implications. LIMUC provides the most clinically actionable evidence in Chapter 3: supervised domain-tuned pipelines substantially outperform the zero-shot baseline on ordinal and clinical-slice metrics under the persisted evaluation configuration. The practical implication for Chapter 4 is direct: UC severity generation must preserve ordinal structure, maintain clinically meaningful threshold behavior (especially specificity), and avoid relying on raw zero-shot generation without safeguards.
3.5 Cross-Dataset Synthesis and Findings
This section synthesizes dataset-wise results into thesis-level findings tied to the research questions. Because datasets differ in answer space and task regime, cross-dataset comparisons are used primarily to identify consistent reliability hierarchies and failure modes, not to claim a single global ranking.
3.5.1 Comparative Reliability: Constrained vs Zero-Shot
Dataset	Strongest constrained/tuned result	Zero-shot/open baseline result	Absolute gap
HyperKvasir	resnet50_supervised acc 0.8789	blip2_zero_shot_clip acc 0.0638	-0.8151
ImageCLEF MEDVQA-GI 2023	vilt_finetune acc 0.9089	qwen_projected acc 0.0670	-0.8419
Kvasir-VQA yes/no subset	resnet_gru_m1 acc 0.9865	blip2_zeroshot_yesno acc 0.8939	-0.0926
LIMUC severity	finetune_resnet50 acc 0.7539	vlm_zero_shot_mayo acc 0.5486	-0.2052
Kvasir-VQA-x1 generative	medgemma_lora token-F1 0.5085	qwen_zeroshot token-F1 0.1728	+0.3357 (adaptation gain)
           Table 3.20 Reliability Gap Snapshot Across Datasets
The repository evidence consistently indicates that constrained/supervised pipelines remain the reliability baseline for structured GI tasks, while raw zero-shot generation is weaker and more format-unstable. The Kvasir-VQA-x1 row summarizes a generative overlap diagnostic (token-F1) rather than accuracy; it shows that adaptation improves overlap fidelity but does not resolve answer-space governance and grounding issues identified in the dataset-wise analysis.
Result for RQ2. Under the persisted evidence in this repository, constrained/supervised methods remain the most reliable core pathway for GI MedVQA, and zero-shot transfer is consistently weaker under strict evaluation conventions.
3.5.2 Dominant Failure Modes (RQ3)
Failure mode	Evidence in this chapter	Practical implication
Head-tail imbalance collapse	HyperKvasir rare recall near zero for multiple models	aggregate accuracy can mask clinically important misses
Lexical drift / non-answer generation	Kvasir-VQA freegen unknown-rate 1.0	requires constrained decoding and output guards
OOV mapping fragility	Kvasir-VQA-x1 mapped VLM acc 0.5616 but macro-F1 0.0058 with OOV 0.973	mapped accuracy alone can be misleading
Question-family brittleness	ImageCLEF: procedure/attribute families collapse for zero-shot raw/projected	per-family reporting is mandatory
Clinical threshold blind spots	LIMUC zero-shot remission specificity 0.0	unsafe threshold behavior without supervision
Scenario evidence gap	reformatted tree provides scenario config without scored outputs	scenario-level failure conclusions are deferred until outputs are persisted
            Table 3.21 Observed Failure Taxonomy
These failure modes explain why single-score reporting is insufficient for clinical MedVQA [56]. Some failures are data-driven (long-tail imbalance), others are format-driven (unknown/OOV and mapping fragility), and others are clinically risk-driven (threshold failures and ordinal inconsistency). This taxonomy directly motivates the risk-control and evidence-governance design in later chapters.

    Figure 3.8 — Failure taxonomy tree
3.5.3 Severity Robustness (RQ4)
The LIMUC results provide the strongest empirical support for RQ4. The key conclusion is that severity-oriented MedVQA must be evaluated using ordinal and threshold-aligned metrics, not accuracy alone. Under the persisted evaluation configuration, tuned models achieve stronger ordinal agreement (QWK) and more stable remission-threshold behavior than the zero-shot baseline, while per-class results show that underrepresented moderate-to-severe classes remain the principal bottleneck.
3.5.4 Statistical Stability
Where paired predictions and McNemar diagnostics are persisted and reported (e.g., within specific dataset blocks that include n01/n10/p-values), the evidence indicates that some large observed differences are not plausibly explained by noise-level fluctuations. For dataset blocks where paired tests are not included in the chapter tables, differences are interpreted conservatively as descriptive observed gaps rather than formal significance claims.
3.5.5 Threats to Validity and Boundaries
Threat	Potential bias	Mitigation applied
Cross-dataset task heterogeneity	direct metric comparison may be invalid	comparisons are primarily within dataset/task
Incomplete artifact parity	some runs have training logs but missing validation preds	explicitly marked as unavailable; no fabricated comparisons
Label projection inflation risk	projected text may match labels lexically without semantic correctness	projected scores reported as diagnostic, not final clinical score
Scenario artifact incompleteness	no persisted reformatted scenario predictions/metrics	scenario claims removed from quantitative synthesis
Legacy vs reformatted pipeline differences	potential metric provenance mismatch	chapter constrained to Prototyping_reformat sources only
            Table 3.22 Threats to Validity and Mitigation
The core methodological risk in this chapter is overinterpreting incomplete artifacts. The mitigation strategy is provenance-first: claims are tied directly to persisted tables, and diagnostics (projection/mapping) are clearly labeled as diagnostics rather than clinical correctness evidence.

3.5.6 Positioning Against Broader MedVQA Findings
The empirical patterns observed here align with the broader MedVQA observations reviewed in Chapter 2: constrained or fine-tuned pipelines remain stronger than naive zero-shot generation on structured clinical tasks, and overlap improvements in generative settings do not automatically guarantee grounded or clinically faithful reasoning [57], [60], [61]. Chapter 3’s contribution is not a state-of-the-art claim; it is a reproducible reliability map for GI MedVQA under a single artifact envelope, explicitly identifying format instability, mapping fragility, question-family brittleness, and threshold failures that must be addressed before clinician-facing deployment.
3.6 Position After Chapter 3
Chapter 3 establishes a reproducible empirical baseline for the dissertation:
1.	GI MedVQA performance is strongly task- and format-dependent; closed-label and free-text regimes behave differently and must be evaluated differently.
2.	Under the persisted evidence in this repository, constrained/supervised methods remain the reliability baseline for closed and ordinal clinical tasks.
3.	Naive zero-shot generation is insufficient as a standalone clinical answer path under strict evaluation conventions and ontology requirements.
4.	Severity-focused evaluation must remain central, especially ordinal agreement and clinically meaningful threshold slices (e.g., remission specificity).
5.	Generative capability is valuable only with explicit governance, including constraints/normalization, unknown/OOV handling, abstention behavior, and (in later chapters) evidence-aware grounding for higher-level queries.
These findings motivate Chapter 4, which develops a controlled generative pipeline for UC severity that preserves ordinal reliability and introduces safeguards for output stability, before extending toward evidence-aware answering.

3.9 References
External Sources
[51] Borgli H, Thambawita V, Smedsrud PH, et al. HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. *Scientific Data*, 2020. https://www.nature.com/articles/s41597-020-00622-y
[52] Gautam S, Storas A, Midoglu C, et al. Kvasir-VQA: A Text-Image Pair GI Tract Dataset. arXiv:2409.01437, 2024. https://arxiv.org/abs/2409.01437
[53] Gautam S, Riegler MA, Halvorsen P. Kvasir-VQA-x1: A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy. arXiv:2506.09958, 2025. https://arxiv.org/abs/2506.09958
[54] Hicks S, Storas A, Halvorsen P, de Lange T, Riegler M, Thambawita V. Overview of ImageCLEFmedical 2023 - Medical Visual Question Answering for Gastrointestinal Tract. CEUR Workshop Proceedings Vol-3497, 2023. https://ceur-ws.org/Vol-3497/paper-107.pdf
[55] Polat G, Kani HT, Ergenc I, et al. Labeled Images for Ulcerative Colitis (LIMUC) Dataset. Zenodo, 2022. https://zenodo.org/records/5827695
[56] Hicks SA, Strumke I, Thambawita V, et al. On evaluation metrics for medical applications of artificial intelligence. *Scientific Reports*, 2022. https://www.nature.com/articles/s41598-022-09954-8
[57] Lin S, Kryściński W, Wu D, et al. Medical Visual Question Answering: A Survey. arXiv:2111.10056, 2021. https://arxiv.org/abs/2111.10056
[58] Ben Abacha A, Hasan SA, Datla V, et al. VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019. CEUR Workshop Proceedings Vol-2380, 2019. https://ceur-ws.org/Vol-2380/paper_78.pdf
[59] Sial M, Fatima M, Nawaz K, et al. Path-RAG: Knowledge-Based Explainable Medical VQA with Large Language Models. Proceedings of Machine Learning Research 259, 2025. https://proceedings.mlr.press/v259/sial25a.html
[60] Yan Q, He X, Yue X, Wang XE. Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA. arXiv:2405.20421, 2024. https://arxiv.org/abs/2405.20421
[61] Rieff M, Varma M, Rabow O, et al. SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning. arXiv:2506.21355, 2025. https://arxiv.org/abs/2506.21355
[62] Stidham RW, Liu W, Bishu S, et al. Performance of a Deep Learning Model vs Human Reviewers in Grading Endoscopic Disease Severity of Patients With Ulcerative Colitis. *JAMA Network Open*, 2019;2(5):e193963. https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2733432
[63] Ozawa T, Ishihara S, Fujishiro M, et al. Novel Computer-Aided Diagnosis System for Endoscopic Disease Activity in Patients with Ulcerative Colitis. *Gastroenterology*, 2020;158(8):2150-2157.e3. https://www.gastrojournal.org/article/S0016-5085%2820%2930212-2/fulltext
[64] Yao H, Tewari AK, Morais M, et al. Novel deep learning-based computer-aided diagnosis system for predicting inflammatory activity in ulcerative colitis: a prospective multicentre study. *Gastrointestinal Endoscopy*, 2023;97(2):330-339.e1. https://pubmed.ncbi.nlm.nih.gov/35985375/
[65] Takenaka K, Ohtsuka K, Fujii T, et al. Development and Validation of a Deep Neural Network for Accurate Evaluation of Endoscopic Images From Patients With Ulcerative Colitis. *Journal of Crohn's and Colitis*, 2023;17(4):463-472. https://academic.oup.com/ecco-jcc/article/17/4/463/6762568

Internal Repository Sources
[66] `Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md`
[67] `Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md`
[68] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md`
[69] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md`
[70] `Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md`
[71] `Prototyping_reformat/DatasetAnalysis/Kvasir_SEG/Kvasir_SEG.md`
[72] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/evaluation_comparison/scenarios/scenarios.yaml`
[73] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/evaluation_comparison/phase3_results/summary_uc_phase3.csv`
[74] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/0_dataset_prep/out/visualizations/qa_by_source.csv`
[75] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/0_dataset_prep/out/visualizations/question_type_counts.csv`
[76] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/0_dataset_prep/out/visualizations/answer_type_counts.csv`
