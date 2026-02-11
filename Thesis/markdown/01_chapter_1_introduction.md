# Chapter 1: Introduction

## 1.1 Background and Motivation

Medical Visual Question Answering (MedVQA) is an interdisciplinary field that integrates computer vision (CV), natural language processing (NLP), and clinical medicine. It aims to build artificial intelligence systems that can interpret medical images and answer clinically relevant questions expressed in natural language. In contrast to conventional image classification systems that return only categorical predictions, MedVQA is designed to support question-driven interaction. This interaction model is closely aligned with how clinicians reason in practice, where interpretation is typically guided by targeted questions rather than isolated labels.

MedVQA offers substantial potential for clinical decision support. Its expected benefits include improved diagnostic consistency, reduced physician workload, and wider access to specialist-level interpretation in telemedicine and resource-constrained settings. The long-term objective is not to replace clinical judgment, but to provide an intelligent and transparent assistant that can complement clinical reasoning within existing workflows.

Unlike general-domain VQA, MedVQA faces domain-specific constraints. Medical images often include subtle, overlapping, or low-contrast findings that require specialized interpretation. Clinical questions frequently require multi-step reasoning, in which visual evidence must be combined with biomedical context and task intent. These characteristics make MedVQA both a technically demanding multimodal task and a clinically sensitive reliability challenge.

The technical foundation of MedVQA combines CV-based image representation with NLP-based question understanding and answer synthesis. Early systems were predominantly discriminative, using an image encoder, a question encoder, a fusion module, and a fixed answer set. These systems performed adequately on constrained tasks but showed limited flexibility for nuanced and open-ended clinical questions [1], [2], [3]. Recent work has shifted toward transformer-based multimodal architectures and generative response models, supported by large-scale vision-language pretraining [4], [5]. In parallel, biomedical language models such as BioBERT and BioGPT have improved domain-specific language understanding and generation [6], [7].

Despite this progress, the gap between benchmark performance and reliable clinical utility remains significant. Three persistent tensions define the present landscape:

1. **Expressiveness versus reliability:** Generative models can produce richer answers, but may exhibit lexical drift, weak grounding, or hallucinated content.
2. **Aggregate performance versus clinical robustness:** High overall accuracy can conceal clinically critical failures in minority classes.
3. **General multimodal capability versus medical domain specificity:** Zero-shot transfer from general VLMs often underperforms in specialized clinical settings.

This dissertation is motivated by these tensions and investigates them in the context of gastrointestinal endoscopy, with a specific focus on colonoscopy-oriented MedVQA.

### 1.1.1 From Fixed-Answer Systems to Generative Multimodal Models

The historical evolution of MedVQA provides an important context for this thesis. In early systems, answer generation was operationally equivalent to label selection. This formulation enabled straightforward optimization and evaluation, but it constrained clinically meaningful expression. Complex questions were often reduced to coarse categorical outputs, limiting interpretability and reducing practical value for clinicians.

Transformer-era multimodal models improved cross-modal alignment and enabled free-text response generation. This transition addressed important limitations of earlier systems, particularly weak question-image interaction and restricted output vocabulary. However, it also introduced new risks. In clinical settings, a fluent answer that is weakly grounded in image evidence may be more problematic than a constrained but verifiable answer. Consequently, model expressiveness must be balanced with grounding, calibration, and error control.

Another major shift has occurred at the evaluation level. Instead of reporting performance within a single benchmark only, recent studies increasingly examine cross-dataset behavior and robustness to distributional changes. This is particularly relevant in GI endoscopy, where data heterogeneity in question templates, answer conventions, and visual characteristics can be substantial.

### 1.1.2 Limitations of Aggregate Benchmark Scores in Clinical AI

Benchmark metrics remain necessary, but they are not sufficient for clinical validation. A model can achieve high overall accuracy while failing on rare yet clinically consequential categories. Weighted scores may obscure failures in underrepresented classes, and single-point metrics may not capture the clinical implications of specific error types.

For severity-oriented tasks, ordinal relationships are also important. Confusing adjacent grades is not equivalent to missing severe disease altogether. Therefore, evaluation design in clinical MedVQA should include classwise metrics, imbalance-aware metrics, ordinally sensitive metrics, and scenario-level acceptance logic.

This dissertation adopts that broader evaluation view from Chapter 1 onward. In addition, answer format itself is treated as an evaluable design choice. For certain use cases, label-only output may be sufficient; for others, clinicians may require rationale, confidence estimates, or evidence pointers. A reliable clinical assistant must be assessed on both predictive accuracy and output usability.

## 1.2 Why GI Endoscopy and Colonoscopy

Gastrointestinal (GI) endoscopy is a strategically important target for MedVQA from both clinical and methodological perspectives.

Clinically, colonoscopy supports screening, diagnosis, and follow-up across multiple disease contexts, including colorectal neoplasia and inflammatory bowel disease. Endoscopic interpretation directly affects treatment escalation, surveillance decisions, and procedural reporting. In routine practice, clinicians frequently pose focused questions such as:

- Is an abnormality present?
- How many findings are visible?
- Where is the lesion located?
- What type or morphology is observed?
- What is the likely severity of inflammation?

These question types map naturally to MedVQA task families and are suitable for systematic evaluation.

From a technical perspective, GI endoscopy is challenging. Image quality can vary due to motion blur, specular reflections, debris, variable bowel preparation, and viewpoint variation. Question-answer distributions can be highly imbalanced, with clinically important severe classes often underrepresented. Consequently, colonoscopy MedVQA provides a realistic and stringent setting for evaluating reliability under domain shift and class imbalance.

Recent dataset development makes this thesis direction feasible:

- Kvasir-VQA introduced GI-focused visual question answering at scale (58,849 QA pairs over 6,500 images) [12], [19].
- Kvasir-VQA-x1 expanded complexity and scale (159,549 QA pairs) and introduced transformed/robustness-oriented settings [13].
- ImageCLEF MEDVQA-GI provided shared benchmarking tracks for GI VQA [10], [11].
- HyperKvasir and LIMUC provide broader GI visual context and severity-oriented subsets [15], [18].

These resources enable reproducible analysis, but they do not remove the central challenge: building systems that remain reliable not only on common benchmark patterns, but also on clinically high-risk cases.

### 1.2.1 Colonoscopy Question Families and Their Decision Roles

To ensure methodological clarity, this thesis maps question families to their clinical decision roles.

| Question family | Example | Dominant challenge | Decision role |
|---|---|---|---|
| Presence | "Is there active bleeding?" | False-negative control | Immediate risk awareness |
| Count | "How many polyps are visible?" | Counting under occlusion and partial views | Completeness and burden estimation |
| Location | "Where is the lesion?" | Spatial grounding consistency | Procedural precision and reporting |
| Type/Attribute | "What polyp morphology is present?" | Fine-grained differentiation | Risk stratification and follow-up planning |
| Severity | "What is the likely Mayo score?" | Ordinal robustness under imbalance | Treatment planning and monitoring |

This mapping is used as a design scaffold for subsequent chapters to align model selection and metric selection with practical clinical use.

### 1.2.2 Operational Constraints in Realistic Deployment

Even in pre-deployment research, model design should anticipate operational constraints. These include variable image quality, limited severe-class labels, institutional differences in reporting language, and the need for predictable failure behavior. A clinically acceptable system is not one that never fails; it is one that fails in ways that are detectable, bounded, and actionable.

These constraints motivate a conditional architecture. Closed-set pathways are often preferable where deterministic label control is required. Generative pathways can be used for explanation-oriented outputs under explicit safeguards. Retrieval-backed reasoning should be introduced selectively for questions that require evidence beyond direct visual interpretation.

## 1.3 Clinical Motivation: Ulcerative Colitis Severity as a Flagship Use Case

Ulcerative colitis (UC) severity assessment is a high-value use case for MedVQA in colonoscopy. Endoscopic severity strongly influences treatment decisions and disease monitoring strategies. In clinical practice, severity is commonly represented by standardized scales such as the Mayo Endoscopic Subscore (0-3) and UCEIS (0-8, often mapped to ordinal severity categories). These scales are clinically meaningful, but grading can be difficult in borderline or low-quality frames.

AI support for UC severity has practical relevance because it can improve consistency and reporting efficiency [14], [16], [17]. From a MedVQA perspective, UC severity is also methodologically informative because it combines:

- Fine-grained visual discrimination.
- Ordinal class structure.
- Asymmetric clinical cost of errors.
- Potential demand for explanation and uncertainty communication.

A standard classifier can output a severity label, but a MedVQA interface can additionally provide rationale-oriented responses and controlled follow-up interaction. This thesis adopts that broader perspective while preserving strict emphasis on visual grounding.

### 1.3.1 Workflow-Oriented Motivation

In practical clinical terms, the decision path is closer to:

`Endoscopy frame(s)`  
`-> clinician question`  
`-> visual-grounded answer`  
`-> confidence and uncertainty signal`  
`-> optional evidence-linked follow-up`

This abstraction motivates the thesis design: MedVQA is treated simultaneously as a perception task, a language task, and a workflow integration task.

### 1.3.2 Why UC Severity Is a High-Value Testbed for Chapter 1

UC severity is used as a flagship testbed in Chapter 1 because it concentrates multiple difficult properties in one clinically relevant problem: subtle visual distinctions, class imbalance, ordinal grading, and high consequence of severe under-calling. If a system is not reliable in this setting, its suitability for broader GI decision support is limited.

At the same time, UC severity remains experimentally tractable using currently available datasets and persisted repository artifacts. This combination of clinical significance and methodological tractability makes it an appropriate anchor for the dissertation.

## 1.4 Problem Statement

The central problem addressed in this dissertation is:

**How can a colonoscopy-focused MedVQA pipeline move from benchmark-level answering to clinically grounded, reliable, and evidence-aware decision support?**

This problem is decomposed into four operational gaps:

1. **Data and coverage gap:** Existing datasets provide broad task families but uneven clinical depth and long-tail imbalance.
2. **Model gap:** Zero-shot general VLMs frequently underperform constrained or supervised approaches in GI settings.
3. **Evaluation gap:** Aggregate metrics can conceal clinically unacceptable error patterns.
4. **Workflow gap:** Many systems do not yet connect image-grounded outputs to evidence-oriented clinical reasoning.

### 1.4.1 Evidence from This Repository

Table 1.1 summarizes key empirical signals from persisted repository artifacts.

| Dataset / Task | Strongest saved model result | Key failure signal | Source |
|---|---|---|---|
| ImageCLEF MEDVQA-GI 2023 validation | ViLT accuracy 0.9089, macro-F1 0.5823 | Qwen2.5-VL zero-shot raw accuracy 0.0007; projected 0.0670 | [21] |
| HyperKvasir 23-class test | ResNet50 supervised accuracy 0.8789, macro-F1 0.5943 | Head-tail recall gap remains large; BLIP2 projected accuracy 0.0638 | [20] |
| LIMUC Mayo severity test | Fine-tuned ResNet50 accuracy 0.7539, macro-F1 0.6829, QWK 0.8351 | Zero-shot VLM macro-F1 0.1771, balanced accuracy 0.25 | [24] |
| Kvasir-VQA yes/no subset | ResNet+GRU accuracy 0.9865, macro-F1 0.9650 | Free-generation run shows unknown-rate collapse in persisted artifact | [22] |
| Kvasir-VQA-x1 generative track | MedGemma LoRA token-F1 0.5085 (adaptation gain) | Exact-match remains near zero across persisted modern VLM runs | [23] |

These findings support a staged strategy:

1. Preserve robust supervised/closed-set visual grounding.
2. Add controlled generative capability where it improves clinical interaction value.
3. Introduce retrieval-backed reasoning as a safeguarded extension for higher-level questions.

### 1.4.2 Why This Gap Persists

The gap persists because GI MedVQA is not a single modeling task. It requires robust visual perception under noisy conditions, clinically coherent question understanding, grounded answer generation, and safe behavior under asymmetric risk. Progress in one component does not guarantee end-to-end clinical utility.

A system may be fluent yet weakly grounded, accurate on common classes yet brittle on severe minorities, or stable on fixed prompts yet unstable under linguistic variation. Therefore, this dissertation treats architecture design, evaluation design, and scenario design as interdependent components.

### 1.4.3 Consequences of Leaving the Gap Unresolved

If this gap remains unresolved, systems may appear strong on aggregate benchmarks while underperforming in the situations that matter most clinically. In severity settings, this can lead to under-detection of high-risk cases. In report-oriented settings, it can produce plausible but weakly grounded outputs that increase cognitive burden. In decision-support settings, it can generate recommendations insufficiently tied to evidence.

These consequences justify a conservative methodological stance: prioritize reliability, grounding, and traceability before maximizing output fluency.

## 1.5 Aim and Objectives

### 1.5.1 Aim

To design, evaluate, and document a clinically oriented MedVQA framework for colonoscopy that combines robust visual grounding, question-aware generation, and an evidence-aware extension pathway for higher-risk queries.

### 1.5.2 Objectives

1. Build a reproducible, multi-dataset GI MedVQA benchmark layer using persisted repository artifacts.
2. Quantify comparative reliability of closed-set versus generative approaches under GI-specific conditions.
3. Evaluate UC severity-oriented question answering with clinically meaningful metrics, including imbalance-aware and ordinal-aligned indicators.
4. Define scenario-driven use cases aligned with endoscopist workflows.
5. Specify and test an implementation path for retrieval-augmented, evidence-aware answering without degrading visual-grounded performance.

### 1.5.3 Objective-to-Deliverable Map

| Objective | Dissertation deliverable |
|---|---|
| O1 | Dataset/task profiling and consolidated benchmark tables |
| O2 | Comparative model analysis with failure-mode interpretation |
| O3 | Severity-focused evaluation section with classwise and ordinal metrics |
| O4 | Scenario catalogue with acceptance criteria and error taxonomy |
| O5 | Proposed architecture and phased validation plan for evidence-aware extension |

## 1.6 Research Questions

The dissertation is guided by six research questions that link task design, modeling strategy, and clinical utility.

**RQ1 (Coverage):**  
What clinically relevant colonoscopy question families and answer spaces are represented in current GI MedVQA datasets, and what important gaps remain?

**RQ2 (Comparative reliability):**  
On GI MedVQA tasks, do constrained/discriminative pipelines remain more reliable than zero-shot open-ended VLM generation?

**RQ3 (Failure modes):**  
Which failure modes dominate current GI MedVQA systems: class imbalance, lexical mismatch, localization ambiguity, or domain shift?

**RQ4 (Severity robustness):**  
How reliably can models answer UC severity-oriented questions, especially for underrepresented severe classes?

**RQ5 (Clinical output format):**  
Which output format best supports clinician trust and usability: label only, label plus rationale, label plus confidence, or answer plus retrieved evidence?

**RQ6 (Evidence-aware extension):**  
Can retrieval-augmented reasoning be integrated as a controlled extension without degrading core visual-grounded accuracy?

### 1.6.1 RQ-to-Evaluation Map

| RQ | Primary evidence in this thesis | Candidate metrics |
|---|---|---|
| RQ1 | Dataset profiling and taxonomy analysis | Coverage %, answer cardinality, imbalance ratio |
| RQ2 | Closed-set vs generative comparisons | Accuracy, macro-F1, MCC, EM, token-F1 |
| RQ3 | Error and robustness analysis | Per-class recall, head-tail gap, OOV/unknown rates |
| RQ4 | UC severity-focused experiments | Macro-F1, balanced accuracy, QWK, remission sensitivity/specificity |
| RQ5 | Scenario-driven analysis | Scenario pass rate, high-risk error counts, clinician-facing clarity criteria |
| RQ6 | Retrieval-augmented prototype analysis | Accuracy delta, evidence relevance, factual consistency checks |

### 1.6.2 Research Hypotheses (Working)

- **H1:** In current GI datasets, constrained or supervised decoders will outperform raw zero-shot free generation on core reliability metrics.
- **H2:** Minority severe classes will remain the primary bottleneck even when aggregate accuracy is high.
- **H3:** A conditional evidence-augmented layer can improve interpretability for high-risk questions without compromising visual grounding.

These hypotheses are working assumptions to be tested in later chapters; they are not final claims.

## 1.7 Scope, Assumptions, and Delimitations

### 1.7.1 In Scope

- Colonoscopy and GI endoscopy image-question answering.
- UC severity-oriented analysis as a flagship clinical axis.
- Multi-dataset benchmarking from persisted local artifacts.
- Closed-set and generative answer tracks.
- Design of an evidence-aware extension pathway (RAG/PICO-oriented direction).

### 1.7.2 Out of Scope in This Dissertation Phase

- Prospective real-time deployment in clinical endoscopy units.
- Full regulatory and compliance validation.
- Multi-center randomized outcome trials.
- Production-grade latency optimization for full video streams.

### 1.7.3 Working Assumptions

- Ground-truth annotations are treated as operational references despite known inter-observer variation.
- Persisted repository artifacts provide the reproducible evidence base for current claims.
- Evidence-aware generation is treated as an additive and safeguarded layer.
- High-risk low-confidence outputs require abstention/escalation behavior.

## 1.8 Scenario-Driven Framing

This dissertation adopts a scenario-driven evaluation philosophy. Aggregate benchmark metrics remain important, but they do not fully capture clinical utility. Scenarios are defined to represent realistic clinician questions and asymmetric risk conditions.

### 1.8.1 Core Scenarios

| Scenario ID | Clinical question style | Required output | Primary risk | Why it matters |
|---|---|---|---|---|
| S1 | "What is the likely Mayo severity?" | Ordinal class + short rationale | Under-calling severe disease | Treatment escalation decisions |
| S2 | "Is active bleeding visible?" | Binary answer + confidence | False negatives in high-risk frames | Urgent management context |
| S3 | "How many polyps and where?" | Count + location-aware response | Missed findings or localization error | Procedural completeness and reporting |
| S4 | "Summarize visible findings." | Multi-finding summary | Hallucinated findings | Interpretability and communication quality |
| S5 | "Given severe findings, what evidence-backed options are relevant?" | Answer + evidence pointer | Weakly grounded management guidance | Decision-support extension |

Each scenario is designed to expose a distinct failure mode. S1 and S2 prioritize risk-sensitive reliability. S3 emphasizes spatial grounding and structured output fidelity. S4 evaluates controlled generation quality. S5 evaluates the boundary between visual interpretation and evidence-oriented reasoning.

### 1.8.2 Scenario Acceptance Logic (Placeholder)

`Question received`  
`-> visual-grounded answer candidate`  
`-> confidence and consistency checks`  
`-> if high-risk and low confidence: abstain/escalate`  
`-> if management-style query: activate retrieval evidence layer`  
`-> return clinician-facing response`

### 1.8.3 Scenario-to-Metric Mapping

| Scenario | Priority metrics |
|---|---|
| S1 | Macro-F1, QWK, severe-class recall |
| S2 | Sensitivity/recall, NPV, abstention behavior |
| S3 | Count error (MAE/RMSE), location correctness slices |
| S4 | Token-F1/ANLS + hallucination audit |
| S5 | Answer correctness + evidence relevance/consistency checks |

In later chapters, this mapping supports a tiered acceptance model:

1. **Technical pass:** baseline quantitative criteria are met.
2. **Scenario pass:** performance is acceptable for the scenario-specific risk profile.
3. **Workflow pass:** output format supports clinician interpretation and action.

## 1.9 Conceptual Framework and System Flow

High-quality technical dissertations typically present Chapter 1 with explicit conceptual architecture, research-design mapping, and chapter dependencies. This chapter follows that pattern while retaining text placeholders where final graphics are pending.

### 1.9.1 End-to-End Conceptual Flow (Text Form)

`Clinical Image(s) + Clinician Question`  
`-> question intent parsing (task type, entities, risk level)`  
`-> visual feature extraction (global + localized cues)`  
`-> cross-modal reasoning (closed-set or generative path)`  
`-> answer candidate + confidence estimation`  
`-> conditional retrieval layer for evidence-oriented queries`  
`-> final response (answer, rationale, confidence, evidence pointer)`

### 1.9.2 Figure 1.1 Placeholder: Clinician-Centric MedVQA Pipeline

**Figure 1.1 (placeholder):** *Clinician-Centric MedVQA Pipeline for Colonoscopy*

Recommended blocks:

- Input block: frame(s), metadata, question.
- NLP block: intent classification and clinical entity extraction.
- Vision block: encoder plus localization cues.
- Fusion block: multimodal reasoning.
- Output block A: closed-set answer.
- Output block B: generative answer.
- Safety block: calibration and abstain gate.
- Retrieval block (conditional): evidence grounding for management-style queries.

### 1.9.3 Figure 1.2 Placeholder: Data-to-Decision Funnel

**Figure 1.2 (placeholder):** *Data-to-Decision Funnel*

`Raw datasets` -> `curation and harmonization` -> `task families` -> `model families` -> `evaluation and failure analysis` -> `scenario validation` -> `decision-support readiness`.

### 1.9.4 Figure 1.3 Placeholder: Research Design Map

**Figure 1.3 (placeholder):** *Research Questions to Methods to Metrics to Outputs*

`RQ layer`  
`-> dataset/method layer`  
`-> metric layer`  
`-> scenario acceptance layer`

### 1.9.5 Figure 1.4 Placeholder: Dissertation Dependency Map

**Figure 1.4 (placeholder):** *Chapter dependency structure*

`Chapter 1 (problem framing)`  
`-> Chapter 2 (scoping review and gap analysis)`  
`-> Chapter 3 (investigating existing VQA techniques across GI-endoscopy datasets)`  
`-> Chapter 4 (proposed pipeline development)`  
`-> Chapter 5 (PICO-oriented use-case instantiation)`  
`-> Chapter 6 (conclusions and future work)`

### 1.9.6 Risk-Control Matrix

**Table 1.2:** *Risk controls for GI MedVQA system design*

| Risk | Failure example | Control strategy |
|---|---|---|
| Class imbalance | Severe UC under-detected | Cost-sensitive training, minority sampling, classwise monitoring |
| Lexical mismatch | Free text not mappable to clinical label space | Constrained decoding and lexical projection safeguards |
| Hallucination | Non-visible finding asserted | Visual-grounding checks, contradiction tests, abstain option |
| Overconfidence | High confidence on incorrect severe class | Calibration, thresholding, uncertainty-triggered escalation |
| Context gap | Image answer lacks decision context | Conditional retrieval-backed evidence layer |

## 1.10 Contributions Claimed in This Dissertation

At this stage, the dissertation claims the following contributions:

1. A consolidated GI MedVQA evidence base across multiple datasets and task formulations using reproducible local artifacts [20]-[24].
2. Empirical clarification that naive zero-shot VLM transfer is currently insufficient for dependable GI MedVQA performance in this repository setting.
3. A severity-focused framing that aligns model evaluation with clinically meaningful endpoints, including macro-F1, balanced metrics, remission slices, and QWK in UC-oriented tasks.
4. A scenario-driven evaluation framework that complements leaderboard metrics with clinically interpretable acceptance criteria.
5. A concrete design pathway for integrating retrieval-augmented, evidence-aware answering as a controlled extension of visual-grounded VQA.

### 1.10.1 Contribution Boundaries

This dissertation does **not** claim immediate real-time clinical deployment readiness. The contributions are methodological, empirical, and design-oriented, and they establish the basis for subsequent translational validation.

## 1.11 Chapter-by-Chapter Summary

- **Chapter 1 (this chapter):** establishes background, motivation, problem decomposition, research questions, conceptual framework, and contribution boundaries.
- **Chapter 2:** provides a structured scoping review of MedVQA and GI endoscopy literature, with explicit gap extraction.
- **Chapter 3:** investigates existing VQA techniques across GI-endoscopy datasets (HyperKvasir, Kvasir-VQA, Kvasir-VQA-x1, ImageCLEF MEDVQA-GI 2023, LIMUC, and supporting Kvasir-SEG analyses), including comparative and failure-mode analyses.
- **Chapter 4:** develops the proposed dataset-model-RAG-fine-tuning pipeline with implementation detail.
- **Chapter 5:** instantiates PICO-oriented and scenario-based use cases with objective-oriented prompt and evidence design.
- **Chapter 6:** synthesizes findings, revisits research questions, discusses limitations, and defines future work.

## 1.12 Chapter 1 Figure and Table Checklist (for Final Layout)

To align with strong dissertation formatting practice, Chapter 1 should include:

1. **Figure 1.1:** Clinician-centric MedVQA pipeline.
2. **Figure 1.2:** Data-to-decision funnel.
3. **Figure 1.3:** Research design map (RQs -> methods -> metrics).
4. **Figure 1.4:** Chapter dependency map.
5. **Table 1.1:** Consolidated empirical gap evidence from local artifacts.
6. **Table 1.2:** Risk-control matrix.
7. **Table 1.3:** RQ-to-evaluation map.
8. **Table 1.4:** Scenario definitions and metric priorities.

If final graphics are not yet ready, captioned placeholders and text-flow diagrams should remain to preserve chapter coherence and avoid structural gaps during final formatting.

## 1.13 References (Chapter 1)

### External Literature

[1] Lau JYC, Gayen S, Ben Abacha A, et al. *A dataset of clinically generated visual questions and answers about radiology images*. Scientific Data, 2018. https://www.nature.com/articles/sdata2018251  
[2] He X, Zhang Y, Mou L, et al. *PathVQA: 30000+ Questions for Medical Visual Question Answering*. arXiv:2003.10286, 2020. https://arxiv.org/abs/2003.10286  
[3] Liu B, Zhan L, Wu X, et al. *SLAKE: A semantically-labeled knowledge-enhanced dataset for medical visual question answering*. arXiv:2102.09581, 2021. https://arxiv.org/abs/2102.09581  
[4] Li C, Wong C, Zhang S, et al. *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day*. arXiv:2306.00890, 2023. https://arxiv.org/abs/2306.00890  
[5] Li J, Li D, Savarese S, Hoi SCH. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. arXiv:2301.12597, 2023. https://arxiv.org/abs/2301.12597  
[6] Lee J, Yoon W, Kim S, et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. arXiv:1901.08746, 2019. https://arxiv.org/abs/1901.08746  
[7] Luo R, Sun L, Xia Y, et al. *BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining*. arXiv:2210.10341, 2022. https://arxiv.org/abs/2210.10341  
[8] Li R, Liu L, Xie Q, et al. *Towards Medical Visual Question Answering with Large Multimodal Models*. arXiv:2501.07109, 2025. https://arxiv.org/abs/2501.07109  
[9] Dong W, Shen S, Han Y, et al. *Generative Models in Medical Visual Question Answering: A Survey*. Applied Sciences, 2025. https://doi.org/10.3390/app15062983  
[10] Murtaza N, Munsif S, Cuadros M, et al. *Overview of ImageCLEFmedical 2023 - Medical Visual Question Answering for Gastrointestinal Tract*. CEUR-WS Vol-3497, 2023. https://ceur-ws.org/Vol-3497/paper-107.pdf  
[11] ImageCLEF. *ImageCLEFmed VQA 2024 Task Page*. 2024. https://www.imageclef.org/2024/medical/vqa  
[12] Gautam S, Riegler MA, Halvorsen P. *Kvasir-VQA benchmark paper (GI endoscopy VQA)*. arXiv:2409.04556, 2024. https://arxiv.org/html/2409.04556v2  
[13] Gautam S, Riegler MA, Halvorsen P. *Kvasir-VQA-x1: A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy*. arXiv:2506.09958, 2025. https://arxiv.org/abs/2506.09958  
[14] Murino A, Rimondi A. *Automated Artificial Intelligence Scoring Systems for the Endoscopic Assessment of Ulcerative Colitis: How Far Are We from Clinical Application?* Gastrointestinal Endoscopy, 2023. https://doi.org/10.1016/j.gie.2022.10.010  
[15] LIMUC dataset repository (with paper/protocol links). https://github.com/wanghaining/ulcerative_colitis  
[16] CLoE benchmark. arXiv:2506.08652, 2025. https://arxiv.org/abs/2506.08652  
[17] Elkhatib et al. *GPT-4V for BBPS quality grading on HyperKvasir*. BMJ Open Gastroenterology, 2025. https://pubmed.ncbi.nlm.nih.gov/40633642/  
[18] Borgli H, Thambawita V, Smedsrud PH, et al. *HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy*. Scientific Data, 2020. https://doi.org/10.1038/s41597-020-00622-y  
[19] Pogorelov K, Thambawita V, Hicks S, et al. *Kvasir-VQA: A Dataset for Visual Question Answering in Gastrointestinal Endoscopy*. Scientific Data, 2023. https://www.nature.com/articles/s41597-023-01981-y  

### Internal Empirical Sources (This Repository)

[20] `Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md`  
[21] `Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md`  
[22] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md`  
[23] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md`  
[24] `Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md`
