# Chapter 2: Surveying VQA Techniques in Medicine with Application to Colonoscopy

## 2.1 Introduction

Visual Question Answering (VQA) has evolved from a general computer-vision benchmark task into a clinically relevant multimodal research area. In medical VQA (MedVQA), a model must interpret medical imagery, parse a natural-language question, and generate an answer that is not only technically correct but also clinically meaningful. This imposes stricter requirements than general-domain VQA: stronger visual grounding, domain-aware language understanding, robustness under class imbalance, and error behavior that is safe under clinical risk.

For this dissertation, the focus is gastrointestinal (GI) endoscopy with an emphasis on colonoscopy use cases. This domain is clinically important and methodologically difficult. Colonoscopy frames contain artifacts, illumination variation, blur, occlusions by instruments, and subtle lesion morphology. Questions that appear simple in natural language (e.g., "Is there active inflammation?" or "What is the likely severity?") can require fine-grained perception and domain-specific interpretation.

The purpose of this chapter is to provide a rigorous survey that supports design choices in subsequent chapters. Specifically, this chapter:

1. maps the historical evolution of MedVQA model families;
2. reviews dataset and benchmark development, with GI-specific focus;
3. compares technique families against colonoscopy scenario requirements;
4. reviews evaluation practices and their limitations;
5. identifies open gaps that directly motivate the methodology in Chapter 3 and Chapter 4.

The chapter is written in the same formal style as the dissertation introduction and uses standard terminology from computer vision (CV), natural language processing (NLP), and multimodal learning.

---

## 2.2 Scoping Review Design

### 2.2.1 Review Objective

This chapter follows a structured scoping-review approach rather than a strict systematic-review protocol. The objective is methodological coverage and decision support for thesis design, not exhaustive bibliometric counting. Sources were selected to cover:

- foundational MedVQA datasets and methods,
- modern multimodal large model directions,
- GI-specific benchmark and challenge resources,
- explainability and retrieval-grounding directions,
- and evaluation methodology relevant to clinical deployment.

### 2.2.2 Search Window and Source Types

The search window for this chapter is **2018 to February 10, 2026**.

Primary source types:

- peer-reviewed papers and data descriptors,
- official benchmark/challenge pages,
- official challenge overview papers and proceedings,
- and influential preprints where they define active benchmark directions.

### 2.2.3 Query Themes

**Table 2.1. Query Themes Used for the Scoping Review**

| Theme | Example queries |
|---|---|
| General MedVQA foundations | `medical visual question answering dataset`, `VQA-RAD`, `PathVQA`, `SLAKE` |
| Modern model families | `LLaVA-Med`, `Med-Flamingo`, `BLIP-2 medical VQA`, `medical LVLM benchmark` |
| GI-specific resources | `Kvasir-VQA`, `Kvasir-VQA-x1`, `ImageCLEF MEDVQA-GI`, `MediaEval Medico 2025` |
| Evaluation and reliability | `medical AI evaluation metrics`, `MedVQA robustness`, `hallucination medical VQA` |
| Explainability and grounding | `explainable medical VQA`, `multimodal explanation GI VQA`, `retrieval-augmented medical VQA` |

### 2.2.4 Inclusion and Exclusion Logic

**Table 2.2. Inclusion and Exclusion Criteria**

| Type | Criteria |
|---|---|
| Inclusion | Primary technical work on MedVQA methods, datasets, benchmarks, or challenge design |
| Inclusion | GI/endoscopy resources directly usable for colonoscopy-oriented VQA analysis |
| Inclusion | Official benchmark/task pages and official challenge overview papers |
| Inclusion | Recent preprints with direct methodological or benchmark impact (explicitly marked as preprint where relevant) |
| Exclusion | Opinion pieces without technical evidence |
| Exclusion | Sources with no direct relation to visual-question answering in medicine |
| Exclusion | Tertiary summaries when primary sources were available |

### 2.2.5 Core Study Pool

The final chapter synthesis uses a curated pool of foundational and recent sources across datasets, models, evaluation, and challenge design. The resulting pool is intentionally broad enough to cover the field trajectory and narrow enough to remain actionable for thesis methodology.

**Figure 2.1 (placeholder): Scoping review flow**

`Identification -> Screening -> Eligibility -> Included core studies`

---

## 2.3 Evolution of VQA Methods: From General AI to Clinical MedVQA

The technical development of MedVQA follows, with delay, the broader VQA trajectory in AI.

### 2.3.1 Stage A: Early Discriminative Pipelines

Early VQA systems relied on CNN image encoders and RNN question encoders with shallow fusion. In these setups, the final task is usually answer classification from a fixed vocabulary. This structure is data-efficient and operationally stable for constrained question sets, but limited for open-ended clinical answers.

### 2.3.2 Stage B: Attention and Cross-Modal Transformers

Transformer-based multimodal architectures improved cross-modal alignment and reasoning depth. Two-stream and cross-attention approaches such as ViLBERT and LXMERT established a stronger pretraining paradigm for vision-language tasks [R31], [R32]. Large-scale contrastive pretraining (e.g., CLIP) then made transfer learning more scalable [R33].

### 2.3.3 Stage C: Instruction-Tuned Generative Multimodal Models

Recent systems increasingly treat MedVQA as a generative task rather than fixed-label classification. The shift is visible in BLIP-2 style adaptation [R34], LLaVA-style visual instruction tuning [R35], and medical-domain variants such as LLaVA-Med [R6] and Med-Flamingo [R7].

The advantage is richer interaction and explanation-like output. The risk is that fluent generated text can become weakly grounded, lexically mismatched to benchmark labels, or clinically unsafe if not constrained.

### 2.3.4 Stage D: Reliability, Explainability, and Evidence Grounding

The current frontier moves beyond "answer generation" toward clinical trust requirements: grounding, calibration, interpretability, and evidence linkage. This includes benchmark-level work on probing reliability [R11], multimodal in-context robustness [R12], explainability frameworks [R15], and retrieval-based methods in clinical VQA settings [R30].

**Figure 2.2 (placeholder): Historical trajectory of MedVQA**

`CNN+RNN -> Attention fusion -> Multimodal transformers -> Generative MLLMs -> Explainable and evidence-aware systems`

---

## 2.4 Dataset and Benchmark Landscape

Dataset design constrains what models can learn and what claims can be made. This section first reviews broad MedVQA resources, then GI/colonoscopy datasets central to this thesis.

### 2.4.1 Foundational Cross-Domain MedVQA Datasets

**Table 2.3. Foundational and Recent General MedVQA Resources**

| Dataset / Benchmark | Year | Reported scope | Why it matters |
|---|---:|---|---|
| VQA-RAD [R1] | 2018 | Clinician-authored radiology QA | Early high-quality clinician-driven MedVQA benchmark |
| PathVQA [R2] | 2020 | Pathology QA from textbook/digital resources | Extended MedVQA to histopathology |
| SLAKE [R3] | 2021 | Bilingual, semantically labeled medical QA | Added richer semantic structure and multilinguality |
| PMC-VQA [R5] | 2023 | Large-scale visual instruction tuning corpus | Enabled modern generative MedVQA pipelines |
| OmniMedVQA [R8] | 2024 | Large multi-dataset LVLM benchmark | Stress-tests generalization across modalities and anatomy |
| MedBookVQA [R9] | 2025 | Textbook-derived multimodal benchmark | Structured benchmark for broad medical domains |
| MedFrameQA [R10] | 2025 (rev. 2026) | Multi-image reasoning benchmark | Closer to real clinical workflow than single-image QA |

### 2.4.2 GI and Colonoscopy-Focused Resources

**Table 2.4. GI Endoscopy Dataset Ecosystem Relevant to This Thesis**

| Resource | Year | Reported scale | Role in this thesis |
|---|---:|---|---|
| HyperKvasir [R16] | 2020 | 110,079 images + 374 videos | Core GI visual foundation and class diversity |
| Kvasir-Capsule [R17] | 2021 | 117 videos, 4.7M+ frames, 47k+ labeled frames | Robustness and broader GI variability context |
| LIMUC [R37] | 2022 | 11,276 UC images from 564 patients | UC severity and ordinal evaluation anchor |
| ImageCLEF MEDVQA-GI 2023 [R22], [R23] | 2023 | Multi-subtask GI VQA/VQG/VLQA benchmark | First major GI-specific VQA shared-task setup |
| Kvasir-VQA [R18], [R19] | 2024 | 6,500 images, 58,849 QA pairs | Main GI text-image pair benchmark |
| ImageCLEF MEDVQA-GI 2024 [R24] | 2024 | Second-year challenge, synthesis-linked direction | Signaled task broadening around synthetic workflows |
| Kvasir-VQA-x1 [R20], [R21] | 2025 | 159,549 QA pairs with complexity levels and perturbations | Stronger reasoning and robustness benchmark |
| ImageCLEF MEDVQA 2025 [R25] | 2025 | Third-year challenge with synthetic GI integration | Benchmark evolution toward real-synthetic design |
| MediaEval Medico 2025 [R26], [R27] | 2025 | GI VQA + multimodal explanation subtask | Explicit shift toward explainable clinical interaction |

### 2.4.3 Challenge-Level Benchmark Progression

The GI challenge ecosystem has progressed quickly:

- **2023:** introduced GI VQA/VQG/VLQA as a dedicated challenge design [R22], [R23];
- **2024:** expanded into synthetic-image-oriented workflows [R24];
- **2025:** formalized VQA with synthetic-real integration at ImageCLEF [R25] and explainable GI VQA at MediaEval [R26], [R27].

This progression matters because it changes what "state-of-the-art" means. The comparison target is no longer only answer correctness; it increasingly includes explanation quality, visual grounding, and clinical relevance.

### 2.4.4 Internal Dataset Evidence from This Repository

This repository provides dataset-level analysis and model diagnostics directly used in later chapters:

- `Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md` [I1]
- `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md` [I2]
- `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md` [I3]
- `Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md` [I4]
- `Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md` [I5]

These internal reports are important because they connect literature claims to reproducible local evidence.

**Figure 2.3 (placeholder): GI benchmark lineage**

`HyperKvasir + Kvasir-Instrument -> Kvasir-VQA -> Kvasir-VQA-x1 -> ImageCLEF/MediaEval challenge tracks`

---

## 2.5 Survey of Technique Families

This section compares method families with specific attention to colonoscopy VQA requirements.

### 2.5.1 Rule-Based and Heuristic Logic

Rule-based systems encode expert logic through handcrafted criteria (e.g., color thresholds, morphology cues, manually defined score rules). Their strengths are transparency and deterministic behavior. Their weaknesses are brittleness under visual variability and poor scalability to open-ended language.

In colonoscopy, rule-based components can still be useful as auxiliary constraints (for quality filtering or post-hoc checks), but not as full MedVQA systems.

### 2.5.2 Discriminative CNN/RNN and Early Fusion Models

These pipelines encode image and question separately and perform fusion for closed-set prediction. They remain strong baselines for constrained yes/no or categorical tasks where label spaces are stable and deterministic output is preferred.

Limitations include weaker compositional reasoning and poor flexibility for long-form explanatory answers.

### 2.5.3 Transformer-Based Multimodal Fusion

Transformers improved image-question alignment and contextual reasoning through cross-attention. Pretrained multimodal encoders can be adapted to clinical VQA with improved sample efficiency relative to training from scratch.

In GI settings, this family often provides strong reliability on structured answer spaces, especially where question templates and answer ontologies are controlled.

### 2.5.4 Generative MLLM-Based MedVQA

Generative approaches output free-text answers and support richer interaction. Key drivers include instruction tuning and parameter-efficient adaptation. Representative medical systems include LLaVA-Med [R6], Med-Flamingo [R7], and instruction-tuned large-scale resources such as PMC-VQA [R5].

Benefits:

- richer language output,
- support for rationale-like responses,
- easier integration with conversational interfaces.

Risks:

- hallucination,
- weak visual grounding,
- answer-format instability on closed-label evaluations,
- and clinically ambiguous verbosity.

### 2.5.5 Explainable and Grounded MedVQA

Explainability has moved from optional add-on to benchmark-level requirement in GI challenges [R26], [R27]. The goal is not only to answer but to justify and localize reasoning cues in ways clinicians can audit.

Recent work includes explicit multi-component explainability pipelines [R15] and challenge systems combining QA, explanation generation, and localization [R28].

### 2.5.6 Retrieval-Augmented and Evidence-Aware Methods

Retrieval augmentation is increasingly explored to ground answers in relevant examples or external evidence. In clinical shared tasks, lightweight multimodal RAG has shown practical gains in response quality and schema adherence [R30].

For colonoscopy MedVQA, retrieval should be treated as a controlled extension layer rather than a replacement for robust visual grounding.

### 2.5.7 Comparative Family-Level Synthesis

**Table 2.5. Technique Family Comparison for Colonoscopy MedVQA**

| Family | Typical output mode | Key strengths | Main risks | Best-fit scenarios |
|---|---|---|---|---|
| Rule-based / heuristics | Deterministic labels/rules | Transparent, predictable | Brittle under visual variation | Narrow quality-control checks |
| CNN/RNN discriminative | Closed-set labels | Stable on constrained tasks, efficient | Limited compositional reasoning | Binary/attribute tasks with fixed ontology |
| Transformer fusion | Closed-set or short text | Strong image-question alignment | Data and tuning sensitivity | Multi-category constrained QA |
| Generative MLLM | Free text | Rich interaction and explanation potential | Hallucination, lexical drift | Narrative support with safeguards |
| Explainable MedVQA | Answer + rationale/grounding | Improves trust and auditability | Evaluation standardization still maturing | Clinician-facing decision support |
| Retrieval-augmented | Answer + retrieved evidence | Better factual anchoring in some settings | Retrieval error propagates to answer | Evidence-linked higher-level queries |

---

## 2.6 Evaluation Practices in the Literature

A central finding of this survey is that evaluation practice is often the bottleneck, not only model architecture.

### 2.6.1 Classification Metrics vs Clinical Risk

Many MedVQA papers report accuracy and macro-F1. These are necessary but insufficient for clinical decision support. In imbalanced settings, aggregate metrics can obscure severe-class failure.

### 2.6.2 Generative Metrics and Their Limits

BLEU, ROUGE, METEOR, CIDEr, ANLS, token-F1, and exact match are useful for text overlap but do not directly guarantee clinical correctness. A linguistically similar answer can still be clinically wrong.

### 2.6.3 Calibration, Uncertainty, and Significance

Clinical deployment requires uncertainty-aware behavior. Yet many papers omit calibration diagnostics (e.g., expected calibration error) and significance testing. This limits interpretability of reported gains.

### 2.6.4 Challenge Metric Profiles

- ImageCLEF GI tracks include classification and segmentation-oriented metrics (e.g., accuracy/F1/MCC and region metrics where relevant) [R22], [R23].
- MediaEval Medico 2025 includes text-overlap metrics plus expert-assessed clinical relevance and explainability components [R26], [R27].

### 2.6.5 Recommended Metric Bundle for This Thesis

This thesis uses a multi-layer metric strategy:

1. closed-set performance: accuracy, macro-F1, MCC, balanced accuracy;
2. generative overlap (where applicable): BLEU/ROUGE/METEOR, token-F1/ANLS;
3. severity-aware evaluation: QWK, remission sensitivity/specificity for UC slices;
4. reliability diagnostics: confidence intervals, paired tests, calibration where available;
5. scenario-level checks: high-risk error counts and acceptance criteria.

**Table 2.6. Metric Families and Practical Interpretation**

| Metric family | Example metrics | Useful for | Known blind spot |
|---|---|---|---|
| Closed-set classification | Accuracy, macro-F1, MCC, kappa | Deterministic QA evaluation | Can hide minority severe-class failures if reported alone |
| Generative overlap | BLEU, ROUGE-L, METEOR, ANLS, token-F1 | Text quality comparison | Weak proxy for clinical correctness |
| Ordinal/severity | QWK, MAE/RMSE over ordinal labels | Severity grading behavior | Not reported consistently across papers |
| Calibration/uncertainty | ECE, reliability slices | Risk-aware deployment analysis | Still rare in MedVQA reporting |
| Statistical robustness | CIs, McNemar/paired tests | Claim stability and significance | Frequently missing in benchmark papers |
| Expert review / explainability | Clinician relevance ratings, grounding checks | Clinical usability and trust | Higher annotation effort, protocol variability |

---

## 2.7 Scenario-Oriented Technique Suitability for Colonoscopy

A clinically useful survey should map model families to concrete clinical scenarios.

### 2.7.1 Scenario Mapping

**Table 2.7. Scenario-to-Method Suitability Map**

| Colonoscopy scenario | Dominant question type | Preferred method profile | Reason |
|---|---|---|---|
| Finding presence/absence | Binary | Constrained discriminative or transformer classifier | Stable labels, high reliability requirements |
| Counting findings/instruments | Count + structured category | Transformer with constrained decoding | Better visual alignment than shallow fusion |
| Lesion location description | Spatial text/category | Transformer + localization-aware head | Requires spatial grounding consistency |
| UC severity grading | Ordinal category | Fine-tuned visual backbone + constrained QA interface | Ordinal robustness and severe-class control |
| Narrative explanation for clinician | Open text | Generative MLLM with grounding checks | Better usability, but must be constrained |
| Evidence-linked management query | Open text + evidence | Conditional retrieval-augmented generation | Useful only when core visual answer is reliable |

### 2.7.2 Recommended Hybrid Operating Pattern

The survey supports a staged hybrid architecture for GI MedVQA:

`Stage 1: reliable constrained answering`  
`-> Stage 2: controlled explanation layer`  
`-> Stage 3: conditional retrieval/evidence layer`

This pattern balances reliability and expressiveness better than a single-model, fully open-ended strategy.

**Figure 2.4 (placeholder): Hybrid decision-support flow**

`Image + Question -> Core visual grounding module -> (if low-risk structured query) constrained answer`

`Image + Question -> Core visual grounding module -> (if explanation needed) grounded generative module`

`Image + Question -> Core visual grounding module -> (if evidence-heavy query) retrieval-assisted module`

---

## 2.8 2024-2026 Trends in MedVQA Research

Recent literature indicates several important shifts.

### 2.8.1 From Single-Image QA to Multi-Image Reasoning

MedFrameQA shows that multi-image clinical reasoning remains difficult for current MLLMs and exposes a gap between benchmark fluency and longitudinal diagnostic reasoning; notably, the benchmark received an updated arXiv revision in February 2026 [R10].

### 2.8.2 From Benchmark Accuracy to Reliability Stress Testing

ProbMed-style probing demonstrates that high benchmark scores can hide brittle behavior under controlled perturbations [R11]. SMMILE similarly highlights fragility in multimodal in-context learning [R12].

### 2.8.3 Data-Centric Expansion and Synthetic Pipelines

New resources increasingly use automated or semi-automated pipelines to scale MedVQA data. This is seen in MedVLSynther-like generator-verifier approaches [R13] and challenge tracks that explicitly integrate synthetic data [R25].

### 2.8.4 Scaling Unified Medical VLMs

OmniV-Med and related work aim at unified multimodal medical understanding across 2D/3D/video settings [R14]. These models are promising for broad capability, but consistent clinical robustness on specialized GI QA remains an open question.

### 2.8.5 Explainability as a First-Class Objective

GI challenge design in 2025 formalized explanation quality and clinical relevance as evaluation targets, not optional analysis [R26], [R27]. This marks a practical shift toward clinician-facing trust criteria.

---

## 2.9 Evidence Triangulation with Repository Results

To avoid a purely narrative survey, this section triangulates literature findings with local persisted artifacts.

### 2.9.1 Consolidated Signals from Local Benchmarks

**Table 2.8. Repository Evidence Snapshot and Method Implications**

| Dataset/task (local report) | Representative local signal | Method implication |
|---|---|---|
| ImageCLEF MEDVQA-GI 2023 [I4] | ViLT fine-tune val accuracy 0.9089, macro-F1 0.5823; zero-shot Qwen raw near-zero accuracy | Closed-set tuned models remain stronger than raw zero-shot generation in this setting |
| HyperKvasir 23-class [I1] | ResNet50 supervised outperforms saved zero-shot generative baseline by large margin | Robust supervised visual encoders remain critical for GI grounding |
| Kvasir-VQA [I2] | ResNet+GRU yes/no subset reaches high reliability; free generation runs can collapse to unknown outputs | Constrained answer spaces remain highly effective for deterministic clinical sub-questions |
| Kvasir-VQA-x1 [I3] | LoRA improves token-level generative metrics; exact-match remains challenging | Reasoning-rich QA increases complexity; output normalization and grounding are central |
| LIMUC severity [I5] | Fine-tuned ResNet50 leads macro-F1 and QWK; zero-shot VLM underperforms severe-class reliability | UC severity tasks favor domain-tuned supervised pipelines with ordinal-aware evaluation |

### 2.9.2 Interpretation

The local results align with broader literature:

- constrained/fine-tuned pipelines are still the reliability baseline in GI tasks;
- generative models provide flexibility but require strict controls;
- evaluation must emphasize classwise and severity-aware behavior, not only aggregate scores.

---

## 2.10 Key Gaps and Open Problems

### 2.10.1 Data and Annotation Gaps

- Severe-class and rare-finding imbalance remains substantial.
- Cross-center, cross-device robustness evidence is still limited for GI VQA.
- Multi-image/video QA is growing but still comparatively immature.

### 2.10.2 Model and Grounding Gaps

- Open-ended outputs often outpace grounding reliability.
- General-domain zero-shot transfer remains unstable for specialized GI semantics.
- Explainability outputs are not yet standardized across benchmarks.

### 2.10.3 Evaluation and Reproducibility Gaps

- Metric heterogeneity hinders fair cross-study comparison.
- Many studies still lack calibration and statistical confidence reporting.
- Clinical utility is often inferred indirectly from generic NLP overlap metrics.

### 2.10.4 Translational Gaps

- Few studies evaluate full clinician-in-the-loop workflows.
- Escalation policies for uncertain/high-risk outputs are seldom formalized.
- Evidence-aware QA in GI remains promising but early-stage.

**Table 2.9. Gap-to-Thesis Mitigation Map**

| Observed gap | Mitigation in this dissertation |
|---|---|
| Long-tail severe-class weakness | Class-aware and severity-aware evaluation slices (Chapter 3, Chapter 4) |
| Zero-shot instability in GI | Strong supervised and constrained baselines before open generation extensions |
| Metric mismatch with clinical risk | Multi-layer evaluation protocol with ordinal and imbalance-aware metrics |
| Explainability without standard criteria | Scenario-specific explanation requirements and clinician-oriented quality checks |
| Weak evidence linkage | Conditional retrieval-augmented extension with guardrails |

---

## 2.11 Chapter Summary and Transition

This chapter surveyed MedVQA techniques with a colonoscopy-oriented lens and synthesized evidence from both external literature and local repository artifacts.

The main conclusions are:

1. MedVQA has moved from fixed-label discriminative models toward generative and explainability-aware multimodal systems.
2. GI benchmark resources have matured rapidly from 2023 onward, enabling more realistic evaluation.
3. For high-risk colonoscopy scenarios, constrained and domain-tuned methods currently provide the most reliable core behavior.
4. Generative and retrieval-enhanced methods are valuable extensions, but only when anchored by robust visual grounding and strict evaluation controls.
5. Scenario-driven, clinically aligned evaluation is essential for translational relevance.

These conclusions motivate Chapter 3, which shifts from literature synthesis to empirical investigation of existing model families using the datasets and persisted outputs available in this repository.

---

## 2.12 Figures and Tables Checklist

### Figures (to prepare in final dissertation format)

1. Figure 2.1: Scoping review flow diagram.
2. Figure 2.2: Method evolution timeline (2018-2026).
3. Figure 2.3: GI dataset lineage and benchmark ecosystem map.
4. Figure 2.4: Hybrid colonoscopy MedVQA decision flow.

### Tables (included in this chapter)

1. Table 2.1: Query themes.
2. Table 2.2: Inclusion and exclusion criteria.
3. Table 2.3: Foundational general MedVQA datasets.
4. Table 2.4: GI dataset ecosystem.
5. Table 2.5: Technique family comparison.
6. Table 2.6: Metric families and interpretation.
7. Table 2.7: Scenario-to-method suitability map.
8. Table 2.8: Repository evidence triangulation.
9. Table 2.9: Gap-to-thesis mitigation map.

---

## 2.13 References

### External Sources

[R1] Lau JJ, Gayen S, Ben Abacha A, et al. *A dataset of clinically generated visual questions and answers about radiology images (VQA-RAD).* Scientific Data, 2018. https://www.nature.com/articles/sdata2018251

[R2] He X, Zhang Y, Mou L, et al. *PathVQA: 30000+ Questions for Medical Visual Question Answering.* arXiv:2003.10286, 2020. https://arxiv.org/abs/2003.10286

[R3] Liu B, Zhan L-M, Xu L, et al. *SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering.* arXiv:2102.09542, 2021. https://arxiv.org/abs/2102.09542

[R4] Lin Z, Zhang D, Tao Q, et al. *Medical visual question answering: A survey.* Artificial Intelligence in Medicine, 2023. https://doi.org/10.1016/j.artmed.2023.102611

[R5] Zhang X, Wu C, Zhao Z, et al. *PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering.* arXiv:2305.10415, 2023. https://arxiv.org/abs/2305.10415

[R6] Li C, Wong C, Zhang S, et al. *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day.* arXiv:2306.00890, 2023. https://arxiv.org/abs/2306.00890

[R7] Moor M, Huang Q, Wu S, et al. *Med-Flamingo: a Multimodal Medical Few-shot Learner.* arXiv:2307.15189, 2023. https://arxiv.org/abs/2307.15189

[R8] Hu Y, Li T, Lu Q, et al. *OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM.* arXiv:2402.09181, 2024. https://arxiv.org/abs/2402.09181

[R9] Yip SL, He S, Nie Y, et al. *MedBookVQA: A Systematic and Comprehensive Medical Benchmark Derived from Open-Access Book.* arXiv:2506.00855, 2025. https://arxiv.org/abs/2506.00855

[R10] Yu S, Wang H, Wu J, et al. *MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning.* arXiv:2505.16964, 2025 (revised 2026). https://arxiv.org/abs/2505.16964

[R11] Yan Q, He X, Yue X, Wang XE. *Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA.* arXiv:2405.20421, 2024. https://arxiv.org/abs/2405.20421

[R12] Rieff M, Varma M, Rabow O, et al. *SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning.* arXiv:2506.21355, 2025. https://arxiv.org/abs/2506.21355

[R13] Huang X, Wang N, Liu H, Tang X, Zhou Y. *MedVLSynther: Synthesizing High-Quality Visual Question Answering from Medical Documents with Generator-Verifier LMMs.* arXiv:2510.25867, 2025 (preprint). https://arxiv.org/abs/2510.25867

[R14] Jiang S, Wang Y, Song S, et al. *OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding.* arXiv:2504.14692, 2025 (preprint). https://arxiv.org/abs/2504.14692

[R15] Nguyen H-D, Dang M-A, Le M-T, Le M-T. *MedXplain-VQA: Multi-Component Explainable Medical Visual Question Answering.* arXiv:2510.22803, 2025 (preprint). https://arxiv.org/abs/2510.22803

[R16] Borgli H, Thambawita V, Smedsrud PH, et al. *HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy.* Scientific Data, 2020. https://www.nature.com/articles/s41597-020-00622-y

[R17] Smedsrud PH, Thambawita V, Hicks SA, et al. *Kvasir-Capsule, a video capsule endoscopy dataset.* Scientific Data, 2021. https://www.nature.com/articles/s41597-021-00920-z

[R18] Gautam S, Storas A, Midoglu C, et al. *Kvasir-VQA: A Text-Image Pair GI Tract Dataset.* arXiv:2409.01437, 2024. https://arxiv.org/abs/2409.01437

[R19] Simula Datasets. *Kvasir-VQA dataset page.* https://datasets.simula.no/kvasir-vqa/

[R20] Gautam S, Riegler MA, Halvorsen P. *Kvasir-VQA-x1: A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy.* arXiv:2506.09958, 2025. https://arxiv.org/abs/2506.09958

[R21] Simula. *Kvasir-VQA-x1 GitHub repository.* https://github.com/simula/Kvasir-VQA-x1

[R22] ImageCLEF. *ImageCLEFmed MEDVQA-GI 2023 task page.* https://www.imageclef.org/2023/medical/vqa

[R23] Hicks S, Storas A, Halvorsen P, de Lange T, Riegler M, Thambawita V. *Overview of ImageCLEFmedical 2023 - Medical Visual Question Answering for Gastrointestinal Tract.* CEUR-WS Vol-3497, paper-107, 2023. https://ceur-ws.org/Vol-3497/paper-107.pdf

[R24] Simula. *ImageCLEFmed-MEDVQA-GI-2024 repository.* https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2024

[R25] ImageCLEF. *ImageCLEFmed MEDVQA 2025 task page.* https://www.imageclef.org/2025/medical/vqa

[R26] MediaEval. *Medico 2025 task page: VQA with multimodal explanations for GI imaging.* https://multimediaeval.github.io/editions/2025/tasks/medico/

[R27] Gautam S, Thambawita V, Riegler M, Halvorsen P, Hicks S. *Medico 2025: Visual Question Answering for Gastrointestinal Imaging.* arXiv:2508.10869, 2025. https://arxiv.org/abs/2508.10869

[R28] Safwan I, Shaikh MA, Haaris M, Khan R, Tahir MA. *Multi-Task Learning for Visually Grounded Reasoning in Gastrointestinal VQA.* arXiv:2511.04384, 2025 (preprint). https://arxiv.org/abs/2511.04384

[R29] Yim W-w, Ben Abacha A, Yetisgen M, Xia F. *Overview of the MEDIQA-WV 2025 Shared Task on Woundcare Visual Question Answering.* ClinicalNLP 2025. https://aclanthology.org/2025.clinicalnlp-1.3/

[R30] Karim AHMR, Uzuner O. *MasonNLP at MEDIQA-WV 2025: Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA.* ClinicalNLP 2025. https://aclanthology.org/2025.clinicalnlp-1.10/

[R31] Lu J, Batra D, Parikh D, Lee S. *ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.* arXiv:1908.02265, 2019. https://arxiv.org/abs/1908.02265

[R32] Tan H, Bansal M. *LXMERT: Learning Cross-Modality Encoder Representations from Transformers.* arXiv:1908.07490, 2019. https://arxiv.org/abs/1908.07490

[R33] Radford A, Kim JW, Hallacy C, et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* arXiv:2103.00020, 2021. https://arxiv.org/abs/2103.00020

[R34] Li J, Li D, Savarese S, Hoi SCH. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* arXiv:2301.12597, 2023. https://arxiv.org/abs/2301.12597

[R35] Liu H, Li C, Wu Q, Lee YJ. *Visual Instruction Tuning (LLaVA).* arXiv:2304.08485, 2023. https://arxiv.org/abs/2304.08485

[R36] Hicks SA, Strumke I, Thambawita V, et al. *On evaluation metrics for medical applications of artificial intelligence.* Scientific Reports, 2022. https://www.nature.com/articles/s41598-022-09954-8

[R37] Polat G, Kani HT, Ergenc I, et al. *Labeled Images for Ulcerative Colitis (LIMUC) Dataset.* Zenodo, 2022. https://zenodo.org/records/5827695

[R38] Murino A, Rimondi A. *Automated artificial intelligence scoring systems for the endoscopic assessment of ulcerative colitis: How far are we from clinical application?* Gastrointestinal Endoscopy, 2023. https://pubmed.ncbi.nlm.nih.gov/36509572/

[R39] Hashash JG, Farraye FA, Wang Y, et al. *Inter- and Intraobserver Variability on Endoscopic Scoring Systems in Crohn's Disease and Ulcerative Colitis: A Systematic Review and Meta-Analysis.* Inflammatory Bowel Diseases, 2024. https://pubmed.ncbi.nlm.nih.gov/38547325/

[R40] Lee J, Yoon W, Kim S, et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining.* Bioinformatics, 2020. https://pubmed.ncbi.nlm.nih.gov/31501885/

[R41] Luo R, Sun L, Xia Y, et al. *BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining.* 2022. https://arxiv.org/abs/2210.10341

### Internal Empirical Sources (This Repository)

[I1] `Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md`

[I2] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md`

[I3] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md`

[I4] `Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md`

[I5] `Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md`
