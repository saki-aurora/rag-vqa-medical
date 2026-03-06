# Final Submission Execution Checklist (March 6, 2026)

## 0) Source of Truth Lock

Use these as authoritative chapter sources before any final citation pass:

1. Chapters 1-3: `Thesis/ThesisFinal_v5.docx` (existing Word thesis body)
2. Chapter 4: `Thesis/markdown/04_chapter_4_consolidated_master.md`
3. Chapter 5: `Thesis/markdown/05_chapter_5_genai_wrapper_pico.md`
4. Chapter 6: `Thesis/markdown/06_chapter_6_conclusions_and_future_research.md`
5. References list: `Thesis/markdown/references_word_numbered.txt`

## 1) Non-Negotiable Execution Order

1. Merge Chapter 4, then Chapter 5, then Chapter 6 into `ThesisFinal_v5.docx`.
2. Clean front matter/template leftovers.
3. Remove all placeholders/drafting artifacts.
4. Update Chapter 1 roadmap and contribution phrasing to final structure.
5. Rebuild citations and bibliography only after text is frozen.
6. Update TOC/List of Tables/List of Figures fields.
7. Final abstract and Chapter 1 contribution polish in final tense.

## 2) Word Fix Targets (Find/Replace Checklist)

Apply these in the Word thesis document:

1. Replace old roadmap mentions:
   - `Chapter 7: synthesizes conclusions, revisits research questions, states limitations, and defines future work.`
   - with: `Chapter 6: consolidates conclusions, closes the research questions, states limitations, and defines future research directions.`
2. Replace contribution lead-in:
   - `At this stage, the dissertation claims the following contributions:`
   - with: `This dissertation makes the following contributions:`
3. Remove Chapter 3 drafting placeholder if present:
   - `[if applicable: integrating them into a broader VQA framework or combining with clinical text data – placeholder for chapter 5 intentions]`
4. Remove acknowledgements template text:
   - `This is an optional page. Use your choice of paragraph style for text on this page (1_Para shown here).`
5. Resolve front matter brackets/placeholders:
   - `[Sarthak Kaushik]`
   - `[Department of Computer Science]`
   - `[Faculty of Computer Science]`
   - `[April 2026]`
   - bracketed committee role text placeholders (if still bracketed)
6. Fix copyright/date consistency:
   - align `© ... 2025` with your final submission year format.

Precomputed stale-artifact scan:
`Thesis/markdown/THESIS_V5_STALE_ARTIFACTS_AUDIT_20260306.md`

## 3) Chapter 6 Defense Upgrade (Now Included)

Chapter 6 now includes an examiner-facing synthesis matrix:

- Section: `6.3.3 Examiner-Facing RQ Synthesis Table`
- Table: `Table 6.2. Research Question Closure Matrix`
- Mapping: `RQ -> answering chapter(s) -> key evidence -> bounded final claim`

Use the updated file directly:
`Thesis/markdown/06_chapter_6_conclusions_and_future_research.md`

## 4) Citation Pass Rules (After Text Freeze)

1. Do not renumber citations before all chapter text is final.
2. After merge, run one global citation order pass across Chapters 1-6.
3. Ensure in-text numeric citations map to entries in:
   `Thesis/markdown/references_word_numbered.txt`
4. Current master reference file length is 107 entries; if new in-text citations are added, append entries sequentially.

## 5) Final Field Update in Word

After all text and references are frozen:

1. Update Table of Contents.
2. Update List of Tables.
3. Update List of Figures.
4. Verify caption styles are applied to every figure/table so lists populate.
5. Confirm no residual message like `No table of figures entries found.` remains.

## 6) Already Applied in Markdown Sources

1. `01_chapter_1_introduction.md`: moved to 6-chapter architecture and final-tense contribution lead-in.
2. `02_chapter_2_survey_of_vqa_techniques.md`: removed Chapter 7 dependency mention.
3. `06_chapter_6_conclusions_and_future_research.md`: added the RQ closure synthesis table.
