```
# Sources and Third-Party Attributions

This document lists all third-party models, software libraries, datasets, research references, and specifications used in the `incubrix_translation_qc` project, along with their respective licenses, commercial-use status, and attribution obligations.

---

## 1. Open Source Machine Learning Models

| Resource / Checkpoint | Creator / Organization | License | Commercial Use Status | Attribution & Usage Notes |
| --- | --- | --- | --- | --- |
| **`facebook/nllb-200-distilled-600M`** | Meta AI | CC-BY-NC-4.0 | Non-Commercial / Evaluation Only | Primary distilled multilingual translation model. Requires attribution to Meta AI.

 |
| **`Helsinki-NLP/opus-mt-*`** (MarianMT family) | Helsinki-NLP / OPUS-MT | CC-BY-4.0 | Permissive (Allowed with attribution) | Language-pair translation engines. Requires retention of original copyright notice.

 |
| **`facebook/fasttext-language-identification`** (`lid.176.bin`) | Meta AI | CC-BY-SA-3.0 | Permissive with ShareAlike | Used for independent Quality Control language identification.

 |

---

## 2. Core Libraries and Frameworks

| Library | Version | License | Commercial Use Status | Purpose |
| --- | --- | --- | --- | --- |
| **`transformers`** | `~4.38.0` | Apache-2.0 | Permissive | Hugging Face model loading and inference runtime. |
| **`torch`** | `~2.2.0` | BSD-3-Clause | Permissive | PyTorch CPU tensor computation backend. |
| **`sentencepiece`** | `~0.2.0` | Apache-2.0 | Permissive | Subword tokenizer for NLLB and MarianMT models. |
| **`sacremoses`** | `~0.1.0` | MIT | Permissive | Moses tokenization helper for MarianMT models. |
| **`fasttext-wheel`** / **`fasttext`** | `~0.9.2` | MIT | Permissive | Python bindings for FastText Language Identification. |
| **`typer`** | `~0.9.0` | MIT | Permissive | Command-line interface framework (`src/cli.py`). |
| **`fastapi`** | `~0.109.0` | MIT | Permissive | REST API web server (`src/api/server.py`). |
| **`uvicorn`** | `~0.27.0` | BSD-3-Clause | Permissive | ASGI server implementation for FastAPI. |
| **`pydantic`** | `~2.6.0` | MIT | Permissive | Data validation and schema definitions for API/CLI. |
| **`pytest`** | `~8.0.0` | MIT | Permissive | Test suite execution and validation engine. |
| **`pytest-cov`** | `~4.1.0` | MIT | Permissive | Test coverage generation tool. |
| **`pyyaml`** | `~6.0.1` | MIT | Permissive | Routing configuration parser (`config/routing_table.yaml`). |

---

## 3. Datasets and Test Inputs

| Dataset / File | Source / Origin | License | Usage Notes |
| --- | --- | --- | --- |
| **`data/test_matrix_50.json`** | Synthetic / IncuBrix Test Spec | Public Domain / CC0 | Benchmark matrix containing 50 test cases across 10 target languages.

 |

---

## 4. Scientific Papers and Reference Specs

* **NLLB-200 Paper:** Costa-jussà, M. R., et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. arXiv:2207.04672.
* **OPUS-MT Paper:** Tiedemann, J., & Thottingal, S. (2020). *OPUS-MT – Building open translation services for the World*. EAMT 2020.
* **FastText LID Paper:** Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). *Bag of Tricks for Efficient Text Classification*. arXiv:1607.01759.
* **Assessment Specification:** *IncuBrix Track 04 Candidate Project Assessment - Multilingual Translation with Independent Quality Control*

---

## 5. Compliance and Licensing Summary

* **Non-Commercial Model Constraints:** The `nllb-200-distilled-600M` model weights are governed by Meta's CC-BY-NC-4.0 license and are strictly used for evaluation and non-commercial assessment purposes.

* **No Paid Generation APIs:** This project relies entirely on open-source, open-weight models executed locally on CPU. No paid external services, proprietary APIs, or non-free compute platforms were used.
```