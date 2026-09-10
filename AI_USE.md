# Disclosure of AI Tool Usage

 This document discloses the specific tools and scope of AI assistance utilized during the development of the `incubrix_translation_qc` project.

---

 ## 1\. Tool Allocation & Responsibilities

 - **Gemini:** Used as the primary technical partner for model research, including selection of `nllb-200-distilled-600M` and `MarianMT` checkpoints, system architecture design, dynamic engine routing strategies, regex masking rules, and core Python logic implementation in `src/`.
- **ChatGPT:** Used specifically for drafting, structuring, proofreading, and formatting project documentation and Markdown (`.md`) files, including `README.md`, `SOURCES.md`, and `AI_USE.md`.

---

 ## 2\. Implementation Breakdown

 ### Model & Architecture Guidance — Gemini

 - Evaluated lightweight, CPU-compatible open-weight models and fallback strategies.
- Formulated entity-preservation patterns for URLs, numbers, and Do-Not-Translate placeholders.
- Outlined Quality Control (QC) anomaly detection rules, including repetition-loop detection and length-ratio thresholds.
- Refined CLI (`Typer`) and API (`FastAPI`) execution commands.
- Assisted with architectural decisions related to multilingual translation and independent QC.

 ### Documentation & Markdown Setup — ChatGPT

 - Standardized Markdown formatting across repository documentation.
- Structured table representations for attribution and licensing information.
- Organized directory-tree and project documentation sections.
- Proofread and improved documentation clarity and consistency.
- Assisted with formatting `README.md`, `SOURCES.md`, and `AI_USE.md`.

---

 ## 3\. Review & Ownership

 All AI-assisted architectural decisions, model choices, and code implementations were manually reviewed, integrated, and validated locally through unit and integration testing.

 AI tools were used as development assistance and did not replace manual review, testing, or engineering judgment.

 I retain full understanding, traceability, and ownership of the code and implementation contained in this repository.