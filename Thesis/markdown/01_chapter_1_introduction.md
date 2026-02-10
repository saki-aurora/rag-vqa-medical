# Chapter 1: Introduction

## 1.1 Background and Motivation

Medical Visual Question Answering (MedVQA) sits at the intersection of computer vision, natural language processing, and clinical reasoning. Instead of returning only a class label, a MedVQA system receives an image and a clinical question, then produces a direct answer in natural language. This interaction model is attractive for healthcare because it mirrors how clinicians already work: they ask focused questions about findings, severity, location, uncertainty, and possible next actions.

Early MedVQA datasets and models demonstrated feasibility but were mostly small and modality-concentrated, especially in radiology [1], [2], [3]. As large vision-language models (VLMs) matured, the field shifted from fixed-answer classification toward free-text generation and broader multimodal reasoning [4], [5], [9], [10]. At the same time, biomedical language models such as BioBERT and BioGPT expanded domain-specific language competence for clinical text interpretation and generation [6], [7].

Despite this rapid progress, two issues remain unresolved for clinical use:

1. Reliability on real-world, long-tail, high-imbalance data.
2. Clinical faithfulness of answers, especially when models generate free text.

This dissertation addresses those issues in the context of gastrointestinal (GI) endoscopy, with a focus on colonoscopy and inflammatory bowel disease (IBD), especially ulcerative colitis (UC).

## 1.2 Why GI Endoscopy and Colonoscopy

Colonoscopy is a high-value and high-load clinical workflow: it supports cancer prevention, inflammatory disease monitoring, and procedure-guided treatment decisions. AI support in this space has already shown value in detection and classification tasks, but most systems still behave as "silent predictors" that output labels, scores, or overlays with limited conversational interpretability [15], [18].

From a MedVQA perspective, GI endoscopy is also technically hard:

- Visual complexity is high (specular highlights, motion blur, stool/debris, variable illumination, camera motion).
- Question space is diverse (presence, count, location, type, severity, procedure metadata).
- Clinical cost of error is asymmetric (missing severe inflammation is not equivalent to a benign false alarm).

Recent GI-specific resources significantly improved feasibility:

- Kvasir-VQA: 58,849 QA pairs over 6,500 images [13].
- Kvasir-VQA-x1: 159,549 QA pairs with complexity-aware settings [14].
- ImageCLEF MEDVQA-GI challenge tracks for colonoscopy-oriented VQA [11], [12].

These resources enable meaningful benchmarking, but they also expose the remaining gap between benchmark scores and clinically robust reasoning.

## 1.3 Problem Statement

The central problem addressed in this thesis is:

**How can a colonoscopy-focused MedVQA system move from benchmark-level answering to clinically grounded, reliable, and evidence-aware decision support?**

Current limitations observed in literature and confirmed by repository results include:

- Strong performance on frequent, closed-set patterns but weak performance on rare/hard classes.
- Poor zero-shot transfer of general VLMs to clinical GI answer spaces.
- High fragility on high-cardinality attribute and location questions.
- Limited linkage between visual answers and evidence-based reasoning.

### 1.3.1 Evidence from This Repository

Table 1.1 summarizes key empirical observations already available in this repository.

| Dataset / Task | Strongest saved model result | Key failure signal | Source |
|---|---|---|---|
| ImageCLEF MEDVQA-GI 2023 validation | ViLT accuracy 0.9089, macro-F1 0.5823 | Qwen2.5-VL zero-shot raw accuracy 0.0007; projection only 0.0670 | [20] |
| HyperKvasir 23-class test | ResNet50 supervised accuracy 0.8789, macro-F1 0.5943 | Severe head-tail recall gap; BLIP2 projected accuracy 0.0638 | [19] |
| LIMUC Mayo severity test | Fine-tuned ResNet50 accuracy 0.7539, macro-F1 0.6829, QWK 0.8351 | Zero-shot VLM macro-F1 0.1771, balanced_acc 0.25 | [23] |
| Kvasir-VQA yes/no subset | ResNet+GRU accuracy 0.9865, macro-F1 0.9650 | BLIP-VQA free generation unknown-rate collapse in persisted run | [21] |
| Kvasir-VQA-x1 generative track | Token-F1 improves with adaptation (MedGemma LoRA token-F1 0.5085) | Exact match near zero across modern VLMs in persisted artifacts | [22] |

This pattern motivates a hybrid thesis direction:

1. Keep strong visual grounding from robust supervised pipelines.
2. Add question-aware generation only where it adds clinical value.
3. Introduce retrieval/evidence links for high-risk reasoning outputs.

## 1.4 Aim and Objectives

### 1.4.1 Aim

To design and evaluate a clinically oriented MedVQA framework for colonoscopy that combines robust visual understanding with question-aware generation and evidence-oriented answer support.

### 1.4.2 Objectives

1. Build a reproducible multi-dataset GI MedVQA benchmark spanning classification and generative settings.
2. Quantify where closed-set models remain superior and where generative methods can add value.
3. Evaluate UC severity question answering using clinically meaningful metrics (macro-F1, balanced accuracy, QWK, calibration-oriented indicators).
4. Define and test scenario-driven use cases aligned with endoscopist workflow (severity, detection context, report-style QA).
5. Specify an implementation pathway for retrieval-augmented, evidence-aware answering (PICO-compatible direction) for future chapter development.

## 1.5 Research Questions

The thesis is guided by the following research questions.

**RQ1 (Dataset and task coverage):**  
What clinically relevant colonoscopy question types and answer spaces are currently covered by GI MedVQA datasets, and where are the coverage gaps?

**RQ2 (Model family comparison):**  
On GI MedVQA tasks, do constrained/discriminative methods remain more reliable than zero-shot open-ended VLM generation?

**RQ3 (Failure modes):**  
Which error modes dominate current pipelines: class imbalance, lexical mismatch, localization ambiguity, or domain shift?

**RQ4 (Severity-focused reliability):**  
How reliably can models answer UC severity-oriented questions (Mayo/UCEIS-aligned framing), especially for minority severe classes?

**RQ5 (Clinical utility):**  
Which scenario-specific outputs are necessary for endoscopist trust: answer only, answer plus rationale, answer plus confidence, or answer plus evidence?

**RQ6 (Evidence integration pathway):**  
Can retrieval-augmented reasoning be integrated without degrading core visual-grounded answer accuracy?

### 1.5.1 RQ-to-Evaluation Map

| RQ | Primary evidence in this thesis | Candidate metrics |
|---|---|---|
| RQ1 | Dataset profiling and question taxonomy | Coverage %, class cardinality, imbalance ratio |
| RQ2 | Closed-set vs generative benchmarks | Accuracy, macro-F1, MCC, EM, token-F1 |
| RQ3 | Error analysis and confusion slices | Per-class recall, head-tail gap, OOV rate |
| RQ4 | UC-specific evaluations on LIMUC/related subsets | Macro-F1, QWK, remission sensitivity/specificity |
| RQ5 | Scenario-driven qualitative + quantitative evaluation | Scenario success rate, clinically relevant error counts |
| RQ6 | Retrieval-augmented prototype experiments | Answer accuracy delta, citation relevance, factual consistency checks |

## 1.6 Scope, Assumptions, and Delimitations

### 1.6.1 In Scope

- Colonoscopy and GI endoscopy image-question answering.
- UC severity-oriented analysis as a flagship clinical use case.
- Benchmarking across available repository datasets and persisted runs.
- Both closed-set and generative evaluation tracks.
- Design of an evidence-aware extension path (RAG/PICO-oriented).

### 1.6.2 Out of Scope (Current Thesis Stage)

- Full prospective clinical deployment.
- Real-time video stream optimization at production latency.
- Regulatory certification claims.
- Multi-center randomized clinical validation.

### 1.6.3 Working Assumptions

- Ground-truth annotations are treated as operational reference despite known inter-observer variability in endoscopy scoring.
- Persisted artifacts in this repository represent the reproducible basis for current claims.
- Evidence-aware generation is introduced as a controlled extension to visual grounding, not a replacement for it.

## 1.7 Conceptual Framework and System Flow

Best-practice dissertation introductions in technical fields typically include:  
1) problem-context figure, 2) conceptual pipeline figure, 3) RQ-to-method table, and 4) chapter map figure.  
This chapter adopts the same structure and defines placeholders that can be converted into polished figures in the final Word template.

### 1.7.1 High-Level Conceptual Flow (Text Placeholder)

`Clinical Image + Clinician Question`  
`-> Question Intent Parsing`  
`-> Visual Feature Extraction`  
`-> Cross-modal Reasoning`  
`-> (Optional) Evidence Retrieval for High-risk Queries`  
`-> Answer Generation + Confidence + Rationale`  
`-> Clinician-facing Output`

### 1.7.2 Figure Placeholder: End-to-End Pipeline

**Figure 1.1 (placeholder):** *Clinician-Centric MedVQA Pipeline for Colonoscopy*  

Suggested blocks:
- Input layer: endoscopy frame(s), question text.
- Understanding layer: clinical intent + entity extraction.
- Vision layer: encoder and localized evidence signals.
- Reasoning layer: fusion module (closed-set or generative).
- Retrieval layer (conditional): guideline/literature lookup for management-style questions.
- Output layer: answer, confidence, rationale, evidence pointer.

### 1.7.3 Figure Placeholder: Data-to-Decision Funnel

**Figure 1.2 (placeholder):** *From raw GI datasets to clinically usable answers*  

`Dataset curation -> task definition -> model training -> evaluation -> scenario validation -> decision support readiness`

### 1.7.4 Table Placeholder: Risk-Control Strategy

**Table 1.2 (placeholder):** *Risk controls for MedVQA in colonoscopy*

| Risk | Failure Example | Control Strategy |
|---|---|---|
| Class imbalance | Severe UC under-detected | Cost-sensitive training, minority-focused augmentation |
| Lexical mismatch | Free-text output not mapped to label | Constrained decoding, lexical projection rules |
| Hallucination | Non-visible finding asserted | Grounding checks, abstain policy |
| Overconfidence | High confidence on wrong severe class | Calibration and confidence thresholding |
| Context gap | Correct image reading but poor management answer | Retrieval-backed evidence module |

## 1.8 Clinical Scenarios Driving This Thesis

This thesis is scenario-driven rather than metric-only. Scenarios are selected where wrong answers carry practical clinical cost.

| Scenario ID | Clinical question style | Required output | Why it matters |
|---|---|---|---|
| S1 | "What is the likely Mayo severity?" | Class + rationale | Supports treatment escalation decisions |
| S2 | "Are there active bleeding signs?" | Binary answer + confidence | High-risk false negatives must be minimized |
| S3 | "How many polyps and where?" | Count + location-aware response | Supports procedural completeness and reporting |
| S4 | "What findings are visible overall?" | Multi-finding summary | Bridges detection output and narrative interpretation |
| S5 | "Given severe findings, what evidence-backed options exist?" | Answer + retrieved evidence | Links visual interpretation to decision support |

### 1.8.1 Scenario Flow Placeholder

`Scenario question -> model answer -> clinical acceptance check -> failure analysis -> model/prompt/retrieval refinement`

## 1.9 Contributions Claimed in This Dissertation

At the current stage, this dissertation claims the following contributions:

1. A unified GI MedVQA benchmarking foundation across multiple datasets and task formulations using persisted, reproducible artifacts [19]-[23].
2. A clear empirical demonstration that naive zero-shot VLM transfer is insufficient for GI MedVQA reliability in current settings.
3. A severity-focused framing that aligns technical evaluation with clinically meaningful endpoints (including QWK and remission-oriented slices in LIMUC analysis).
4. A scenario-driven design framework that moves beyond leaderboard metrics toward clinician-relevant utility testing.
5. A practical architecture path for integrating retrieval-augmented, evidence-aware reasoning with image-grounded VQA.

## 1.10 Chapter-by-Chapter Summary

- **Chapter 1 (this chapter):** problem context, motivation, RQs, conceptual framework, and contribution framing.
- **Chapter 2:** structured scoping review of MedVQA and GI endoscopy methods, with gap analysis.
- **Chapter 3:** investigation of existing models on current datasets, with failure-mode analysis.
- **Chapter 4:** development of proposed dataset-model-RAG-fine-tuning pipeline.
- **Chapter 5:** PICO-oriented use scenarios and objective-oriented prompt/evidence design.
- **Chapter 6:** conclusions, limitations, and future work.

## 1.11 Chapter 1 Figure and Table Checklist (for Final Thesis Layout)

To support high-quality dissertation formatting, Chapter 1 should contain the following visual artifacts:

1. **Figure 1.1:** Clinician-centric MedVQA pipeline.
2. **Figure 1.2:** Data-to-decision funnel.
3. **Figure 1.3:** Research design map (RQs -> datasets -> models -> metrics -> scenarios).
4. **Figure 1.4:** Dissertation chapter dependency map.
5. **Table 1.1:** Empirical gap summary from existing repository results.
6. **Table 1.2:** Risk-control matrix.
7. **Table 1.3:** RQ-to-evaluation mapping.
8. **Table 1.4:** Scenario definitions and acceptance criteria.

If final graphics are not ready at drafting time, retain captioned placeholders and text-flow diagrams (as done here) to prevent structural gaps in later formatting passes.

## 1.12 References (Chapter 1)

### External Literature

[1] Lau JYC, Gayen S, Ben Abacha A, et al. *A dataset of clinically generated visual questions and answers about radiology images*. Scientific Data, 2018. https://www.nature.com/articles/sdata2018251  
[2] He X, Zhang Y, Mou L, et al. *PathVQA: 30000+ Questions for Medical Visual Question Answering*. arXiv:2003.10286, 2020. https://arxiv.org/abs/2003.10286  
[3] Liu B, Zhan L, Wu X, et al. *SLAKE: A semantically-labeled knowledge-enhanced dataset for medical visual question answering*. arXiv:2102.09581, 2021. https://arxiv.org/abs/2102.09581  
[4] Li C, Wong C, Zhang S, et al. *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day*. arXiv:2306.00890, 2023. https://arxiv.org/abs/2306.00890  
[5] Li J, Li D, Savarese S, Hoi SCH. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. arXiv:2301.12597, 2023. https://arxiv.org/abs/2301.12597  
[6] Lee J, Yoon W, Kim S, et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. arXiv:1901.08746, 2019. https://arxiv.org/abs/1901.08746  
[7] Luo R, Sun L, Xia Y, et al. *BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining*. arXiv:2210.10341, 2022. https://arxiv.org/abs/2210.10341  
[8] Singh A, Tang Y, Fang Z, et al. *MedThink: a clinically-grounded reasoning framework for medical VQA and decision support*. ACL 2025. https://aclanthology.org/2025.acl-long.1399/  
[9] Li R, Liu L, Xie Q, et al. *Towards Medical Visual Question Answering with Large Multimodal Models*. arXiv:2501.07109, 2025. https://arxiv.org/abs/2501.07109  
[10] Dong W, Shen S, Han Y, et al. *Generative Models in Medical Visual Question Answering: A Survey*. Applied Sciences, 2025. https://doi.org/10.3390/app15062983  
[11] Murtaza N, Munsif S, Cuadros M, et al. *Overview of ImageCLEFmedical 2023 - Medical Visual Question Answering for Gastrointestinal Tract*. CEUR-WS Vol-3497, 2023. https://ceur-ws.org/Vol-3497/paper-107.pdf  
[12] ImageCLEF. *ImageCLEFmed VQA 2024 Task Page*. 2024. https://www.imageclef.org/2024/medical/vqa  
[13] Gautam S, Riegler MA, Halvorsen P. *Kvasir-VQA benchmark paper (GI endoscopy VQA)*. arXiv:2409.04556, 2024. https://arxiv.org/html/2409.04556v2  
[14] Gautam S, Riegler MA, Halvorsen P. *Kvasir-VQA-x1: A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy*. arXiv:2506.09958, 2025. https://arxiv.org/abs/2506.09958  
[15] Murino A, Rimondi A. *Automated Artificial Intelligence Scoring Systems for the Endoscopic Assessment of Ulcerative Colitis: How Far Are We from Clinical Application?* Gastrointestinal Endoscopy, 2023. https://doi.org/10.1016/j.gie.2022.10.010  
[16] LIMUC dataset repository (with paper/protocol links). https://github.com/wanghaining/ulcerative_colitis  
[17] CLoE benchmark. arXiv:2506.08652, 2025. https://arxiv.org/abs/2506.08652  
[18] Elkhatib et al. *GPT-4V for BBPS quality grading on HyperKvasir*. BMJ Open Gastroenterology, 2025. https://pubmed.ncbi.nlm.nih.gov/40633642/  

### Internal Empirical Sources (This Repository)

[19] `Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md`  
[20] `Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md`  
[21] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md`  
[22] `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md`  
[23] `Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md`
