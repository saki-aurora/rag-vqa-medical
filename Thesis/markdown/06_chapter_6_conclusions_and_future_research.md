# Chapter 6: Conclusions and Future Research

## 6.1 Chapter Purpose and Position in the Dissertation

This chapter closes the dissertation by consolidating the technical, empirical, and translational findings developed from Chapter 1 through Chapter 5. The work began with a practical clinical AI tension: GI-endoscopy visual question answering (VQA) systems can report encouraging benchmark scores, yet clinical usefulness depends on reliability under class imbalance, robustness under domain shift, explicit uncertainty communication, and evidence-grounded response behavior.

To address this, the dissertation followed a staged methodological arc rather than a single-model optimization arc:

1. define the clinical problem, scope, and research questions (Chapter 1);
2. establish the MedVQA and GI-endoscopy literature and benchmark landscape (Chapter 2);
3. empirically audit existing model families using persisted GI artifacts (Chapter 3);
4. develop a controlled generative UC severity module with frozen reporting boundaries (Chapter 4);
5. integrate severity-compatible, PICO-grounded, citation-aware physician-query support (Chapter 5).

The central conclusion is that clinically meaningful progress in GI MedVQA is achieved through constrained, auditable systems engineering and bounded claims, not through unconstrained generation alone.

A second, equally important conclusion is that the dissertation contributes a *deployment logic* and not only a *model result*. The contribution is therefore not reducible to a single score table. It is a layered decision framework that clarifies when evidence is strong enough for limited clinical decision support and when the output should be escalated, deferred, or rejected. This distinction is essential in medical AI, where overstating generalization is more harmful than reporting conservative gains.

## 6.2 Consolidated Narrative of What Was Demonstrated

Across the full pipeline, five cross-chapter findings are consistent and mutually reinforcing.

1. **Dataset and answer-space structure governs observed reliability.** GI MedVQA resources vary substantially in question families, answer cardinality, supervision style, and label semantics, so aggregate metrics can hide clinically important weaknesses [54]-[57].
2. **Constrained/supervised pipelines remain a necessary reliability anchor.** Chapter 3 repeatedly showed that naive open-ended generation is brittle under GI-specific constraints, especially for underrepresented classes and output-space variability.
3. **Controlled generative adaptation can surpass strong supervised baselines in-domain.** In Chapter 4 internal LIMUC evaluation, the frozen generative `mode1` lane exceeded the supervised anchor on QWK, macro-F1, balanced accuracy, and accuracy while preserving parse compliance [78], [80]-[84].
4. **Domain-shift robustness is not solved by in-domain gains.** Chapter 4 external proxy evaluation showed substantial degradation, so internal improvements must be interpreted as bounded internal validity rather than universal transfer performance.
5. **Workflow-level safeguards are critical for physician-facing use.** Chapter 5 demonstrated that citation linkage, refusal/escalation behavior, uncertainty surfacing, and completion-audit traceability can be operationalized in a reproducible wrapper [93]-[100].

These findings, taken together, support a practical translational posture: optimize reliability layers and safety behavior first, then scale model capability under explicit governance.

The results also reveal a dependency structure across chapters. Chapter 2 established what should be measured, Chapter 3 showed where existing methods break, Chapter 4 tested whether controlled generation can improve severity reliability, and Chapter 5 tested whether improved modeling can be translated into clinician-facing interaction without removing safeguards. This dependency structure is why Chapter 6 conclusions are stronger than a chapter-isolated summary: each chapter de-risks assumptions for the next chapter.

## 6.3 Integrated Answers to Research Questions

Chapter 1 defined six research questions (RQ1-RQ6). Their final evidence-backed answers are summarized below.

| Research Question | Final Answer from Dissertation Evidence |
|---|---|
| **RQ1 (Coverage):** What clinically relevant GI question families and answer spaces are represented in available datasets? | Coverage exists but is uneven. Dominant families (for example yes/no and template-style prompts) are overrepresented relative to complex reasoning and high-risk edge-case queries, which creates evaluation skew and overestimation risk if not controlled [54]-[57]. |
| **RQ2 (Comparative reliability):** Are constrained/discriminative approaches more reliable than naive zero-shot open generation for GI MedVQA? | Yes, in current repository evidence and task settings. Chapter 3 shows constrained/supervised behavior as a recurring reliability anchor across GI datasets. |
| **RQ3 (Failure modes):** Which failure classes dominate current systems? | Class imbalance sensitivity, output-space mismatch, mapping/OOV instability, and domain-shift fragility emerged as dominant recurring failures. |
| **RQ4 (Severity robustness):** How reliably can models handle UC severity-oriented VQA, including severe classes? | Controlled adaptation in Chapter 4 materially improved ordinal and class-balanced reliability internally, but minority-severity challenges and external robustness limitations remain [78], [80]-[84]. |
| **RQ5 (Clinical output format):** Which output style better supports clinician trust and usability? | Structured outputs with citation linkage, uncertainty, and policy-bounded limitations are more defensible for clinician review than unconstrained fluent responses under current evidence [89]-[100]. |
| **RQ6 (Evidence-aware extension):** Can retrieval-grounded reasoning be integrated without compromising core visual-grounded behavior? | Yes, as a modular extension under explicit boundaries. Chapter 5 demonstrates a reproducible PICO-grounded wrapper with auditability and safety controls while preserving constrained claim scope [93]-[100]. |

### 6.3.1 Interpretation of RQ Closure

The RQ closure pattern is intentionally asymmetric: stronger answers were obtained for *reliability characterization* and *pipeline feasibility* than for *clinical deployment readiness*. This is an expected and acceptable outcome for a dissertation that prioritizes reproducibility and claim discipline. The thesis resolves "what works reliably under controlled evidence conditions" and clearly marks "what still requires larger-scale clinical validation."

### 6.3.2 What RQ Closure Means for Clinical Translation

For translation, the most meaningful RQ outcome is not the strongest metric gain; it is the ability to map each claim to an evidence boundary. In practical terms, RQ closure supports the following translational interpretation:

1. the system is suitable for research-grade decision support experiments under controlled supervision;
2. the system is not yet suitable for unsupervised clinical autonomy;
3. future deployment readiness depends more on robustness and governance evidence than on incremental in-domain benchmark gains.

This interpretation keeps the dissertation aligned with medical safety expectations while preserving the value of the demonstrated technical advances.

### 6.3.3 Examiner-Facing RQ Synthesis Table

To support defense/viva readability, Table 6.2 links each research question to its answering chapter evidence and final bounded claim.

**Table 6.2. Research Question Closure Matrix**

| Research question | Primary chapter(s) answering it | Key evidence used | Bounded final claim |
|---|---|---|---|
| RQ1 (coverage of GI MedVQA tasks and answer spaces) | Chapter 2, Chapter 3 | GI dataset/task-family mapping and benchmark audit [54]-[57] | Coverage is present but uneven; evaluation must control for skewed question/answer distributions. |
| RQ2 (constrained vs naive open generation reliability) | Chapter 3, Chapter 4 | Cross-family reliability comparisons and LIMUC internal anchor [78], [80]-[84] | Constrained/supervised pipelines remain a necessary reliability baseline under current GI settings. |
| RQ3 (dominant failure modes) | Chapter 3 | Failure taxonomy across imbalance, lexical mapping, and OOV behavior | Reliability failures are systematic, not random, and should drive architecture constraints. |
| RQ4 (UC severity robustness, including severe classes) | Chapter 4 | Internal multi-seed LIMUC metrics plus class-wise analysis [78], [80]-[84] | Controlled generative adaptation improves internal ordinal and balanced performance, but does not solve external robustness. |
| RQ5 (clinician-facing output form) | Chapter 5 | Wrapper output audits for citations, uncertainty, and refusal behavior [89]-[100] | Structured, citation-aware outputs are more defensible than unconstrained free-form responses. |
| RQ6 (evidence-grounded extension feasibility) | Chapter 5 | PICO extraction, retrieval, and synthesis pipeline with typed contracts [93]-[100] | Retrieval-grounded extension is feasible as a modular layer when bounded by explicit policy and audit controls. |

## 6.4 Final Contributions

The dissertation contributions can be grouped into methodological, empirical, and engineering contributions.

Together, these contributions define a reproducible pattern for multimodal medical AI development: establish boundaries first, optimize within boundaries second, and expose boundary violations explicitly in output behavior.

### 6.4.1 Methodological contributions

1. A reproducibility-first, claim-bounded evaluation strategy across all core chapters.
2. A staged architecture logic that separates severity scoring reliability from physician-query reasoning reliability.
3. A concrete integration framework where controlled generation is paired with structured evidence presentation rather than free-text fluency-first design.

These methodological decisions reduce hidden coupling between model quality and interface behavior. Without this separation, wrapper-level improvements can mask model-level weaknesses and vice versa.

### 6.4.2 Empirical contributions

1. A GI MedVQA reliability map (Chapter 3) that identifies repeatable strengths and failure modes across dataset regimes.
2. A frozen internal UC severity package (Chapter 4) where controlled generative adaptation outperformed a strong supervised anchor on key ordinal and balanced metrics [78], [80]-[84].
3. A frozen PICO-grounded wrapper benchmark (Chapter 5) demonstrating citation-linked synthesis, refusal-aware behavior, and artifact-auditable output generation [93]-[100].

The empirical contribution is therefore cumulative rather than isolated. Chapter 3 defines baseline risk, Chapter 4 demonstrates bounded improvement, and Chapter 5 demonstrates feasibility of evidence-aware interaction under policy constraints.

### 6.4.3 Engineering contributions

1. End-to-end traceability from data preparation through chapter-level artifact synchronization.
2. Typed interfaces and persisted JSON/JSONL contracts for wrapper reproducibility and auditability [89]-[95].
3. Completion-audit gating that enforces chapter text and artifact consistency at freeze time [100].

These engineering contributions matter because reproducibility in medical AI is an operational requirement, not a cosmetic preference. The dissertation artifacts are organized so that evaluation can be replayed, outputs can be inspected, and claim-evidence links can be externally audited.

## 6.5 Revisit of Dissertation Hypotheses

The working hypotheses in Chapter 1 can now be revisited.

1. **H1:** constrained/supervised pipelines outperform naive zero-shot free generation on reliability metrics: **supported** by Chapter 3 and reinforced by Chapter 4 ablation behavior.
2. **H2:** minority severe classes remain a dominant bottleneck: **supported**; gains were achieved in Chapter 4, but severe-class and boundary ambiguity risks persist.
3. **H3:** evidence-augmented extensions can improve interpretability without sacrificing core visual grounding if appropriately constrained: **supported in baseline form** by Chapter 5 wrapper results under explicit scope limits [93]-[100].

The hypothesis outcomes indicate that the dissertation's central technical strategy is directionally validated.

At the same time, support strength differs by hypothesis. H1 is strongly supported within the tested scope, H2 is supported with unresolved minority-class risk, and H3 is supported for baseline wrapper behavior but remains sensitive to retrieval quality and policy calibration. This differentiated interpretation is important to avoid treating all hypothesis outcomes as equally mature.

## 6.6 Alignment to Requested Technical Direction

Excluding timeline planning (intentionally omitted for submission), the implemented work is aligned with the requested technical sequence:

1. Chapter 2 and Chapter 3 were completed around GI VQA, including Kvasir-VQA and Kvasir-VQA-x1 evidence lines.
2. Chapter 4 developed generative UC severity methods with comparative analytics and now includes the requested IEEE/Springer references [105], [106].
3. Chapter 5 implemented the PICO-based GenAI wrapper and includes the requested PICO GenAI IEEE alignment [107].

This means the requested end-to-end technical progression is represented in dissertation chapter form and bibliography linkage.

From a final-pass perspective, this alignment shows that the dissertation now follows the requested sequence from survey to empirical investigation, to generative severity analytics, to PICO-based wrapper design, and finally to synthesis and future work. The sequence is technically coherent and defensible for submission.

## 6.7 Practical Implications for Clinician-Facing AI

Within current boundaries, three practical implications are defensible.

1. **Evidence and uncertainty must be first-class outputs.** Citation-linked claims with explicit uncertainty and limitations are safer than unsupported fluent responses.
2. **Reliability must be assessed at model and workflow levels simultaneously.** A strong scorer is insufficient if downstream retrieval, synthesis, and safety behavior are uncontrolled.
3. **Negative results are operationally useful.** The Chapter 4 `mode2` collapse and external-shift degradation are not failures of the dissertation; they are constraints that prevent overclaiming and guide robust next-step design.

Two additional practice-level implications follow from the same evidence:

4. **Escalation behavior is a feature, not a defect.** Refusal or defer-to-clinician outputs are necessary safety mechanisms when evidence confidence is weak.
5. **Auditability should be designed before deployment pilots.** Retrofitting traceability after interface rollout is costly and weakens reliability claims.

## 6.8 Limitations and Validity Boundaries

Despite meaningful progress, the dissertation is explicitly bounded by the following limitations.

1. Core claims are based on internal/frozen benchmark protocols rather than prospective production deployment.
2. Chapter 5 PICO/retrieval gold subsets are intentionally small and suitable for baseline validation, not broad clinical generalization.
3. Grounding checks include heuristic components and need larger clinician-semantic adjudication studies.
4. Cross-domain robustness remains unresolved, especially under label-space mismatch and distribution shift.
5. The wrapper is a decision-support prototype and does not generate patient-specific treatment execution instructions.
6. Chapter 5 freeze reporting used `has_severity_context=false` to isolate wrapper behavior; full end-to-end severity-injected effect remains a follow-on research question.

These limitations do not negate the contribution; they define the conditions under which the conclusions are valid.

### 6.8.1 Threats to Validity Framing

The limitations above can be interpreted as standard threats-to-validity categories:

1. **Internal validity threat:** measured gains can be influenced by dataset-specific artifacts and frozen protocol assumptions.
2. **External validity threat:** transfer to new institutions, devices, and reporting standards remains uncertain.
3. **Construct validity threat:** benchmark metrics may only partially reflect clinician-perceived utility and safety.
4. **Conclusion validity threat:** small subset studies can inflate variance and reduce confidence in fine-grained comparative claims.

Stating these categories explicitly improves interpretability of the dissertation outcomes and clarifies what follow-on evidence is required.

## 6.9 Future Research Agenda

The future research path emerging from this dissertation is phased and evidence-driven. Rather than proposing a single large expansion, the next stage is best understood as a sequence of connected studies that progressively improve data quality, retrieval-grounded reasoning, and translational validity.

### 6.9.1 Near-term research priorities

In the near term, the most important limitation to address is evidence coverage and annotation depth in the Chapter 5 wrapper setting. The current knowledge base can be broadened with guideline-level and trial-anchored sources so that retrieval behavior is tested against more realistic clinical evidence variation. At the same time, the PICO and retrieval gold subsets can be expanded through multi-annotator adjudication, with inter-annotator agreement reported explicitly as part of dataset-quality evidence.

A second near-term focus is extraction and grounding quality. The present extractor is intentionally recall-oriented, but future versions should improve precision for outcome and severity-anchor fields without materially reducing recall. In parallel, grounding assessment can move beyond lexical overlap by introducing clinician-semantic review pipelines that better reflect whether synthesized claims preserve intended clinical meaning.

### 6.9.2 Mid-term robustness and integration priorities

The mid-term phase is centered on robustness and module integration. A key unresolved question is the measurable contribution of injected severity context, because the frozen Chapter 5 benchmark intentionally used `has_severity_context=false` to isolate wrapper behavior. Controlled studies with severity context enabled can therefore quantify incremental value under fixed evaluation conditions.

Mid-term work also includes stronger retrieval supervision tied to clinician relevance judgments, which should improve ranking quality at low `k` and reduce dependence on broad evidence pooling. In addition, abstention policy can be recalibrated as a joint function of retrieval confidence, severity uncertainty, and safety triggers, so refusal behavior becomes more explicitly risk-calibrated. Finally, moving from frame-level predictions to temporally coherent sequence-level inference is necessary to better represent full-procedure reasoning in colonoscopy workflows.

### 6.9.3 Long-term translational priorities

Long-term progress depends on prospective, multi-center validation under real institutional workflow constraints. Such studies are required to test transportability across device settings, reporting styles, and site-level practice differences that are not represented in single-repository benchmarks.

Equally important is human-factors evaluation of trust calibration, workload effects, and explanation usability, since model utility in clinical settings depends on interaction quality as much as prediction quality. Over the same horizon, governance mechanisms such as drift monitoring, audit cadence, and safety-case maintenance become core system requirements. This naturally leads to progressive alignment with institutional and regulatory expectations for clinical decision-support technologies.

### 6.9.4 Suggested evaluation extensions

To preserve methodological rigor, future evaluations can be extended in four directions: broader confidence-interval reporting with stratified subgroup analysis; paired statistical testing for wrapper-level ablations where paired outputs are available; explicit error-taxonomy accounting that separates safety-triggered refusals from genuine low-evidence abstentions; and reporting templates that present utility, safety, and uncertainty jointly rather than utility alone.

### 6.9.5 Recommended Program of Work

At program level, this agenda can be organized into three work packages. Work Package A (data and annotation) addresses benchmark diversity, adjudication quality, and agreement reporting. Work Package B (model and retrieval) focuses on severity conditioning, retrieval ranking reliability, and calibrated abstention thresholds. Work Package C (human-in-the-loop evaluation) studies clinician usefulness, trust calibration, and failure-recovery behavior in realistic interaction settings.

Framed this way, the future agenda remains academically grounded while still being operationally executable in lab, grant, or multi-institution collaborations.

## 6.10 Final Conclusion

The dissertation demonstrates that progress toward clinically useful GI MedVQA is feasible when design is constrained, evidence-linked, and auditable. The key insight is not that generation should replace classical reliability anchors; it is that generation becomes useful only when embedded inside explicit task boundaries, robust evaluation policy, and workflow-level safety controls.

In practical terms, this thesis contributes a defensible translational path from benchmark-centric VQA to clinician-aware multimodal decision support. It delivers bounded empirical gains, transparent limitations, and a concrete future-work roadmap that can be executed without relaxing claim discipline.

The final implication for future researchers is methodological: in high-stakes multimodal AI, reliability should be treated as a systems property. Model architecture, data regime, prompt/policy constraints, retrieval behavior, and escalation logic must be evaluated together. This dissertation provides a starting blueprint for that integrated evaluation approach in GI-endoscopy MedVQA and UC severity-oriented clinical reasoning support.
