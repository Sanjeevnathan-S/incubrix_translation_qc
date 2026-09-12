# Technical Report: Multi-Engine Translation & Quality Control Pipeline

---

 ## 1\. System Architecture & Component Design

 The system implements a multi-engine machine translation architecture designed for CPU-constrained environments. It integrates two primary model families, an automated Do-Not-Translate (DNT) entity preservation system, and a post-translation Quality Control (QC) validation suite.

```
                  ┌────────────────────────┐
                  │   Input Source Text    │
                  └───────────┬────────────┘
                              │
                  ┌───────────▼────────────┐
                  │ DNT Regex Masking Engine│
                  └───────────┬────────────┘
                              │
                  ┌───────────▼────────────┐
                  │    Language Router     │
                  └─────┬────────────┬─────┘
                        │            │
         Direct / High-Perf          Fallback / Pivoted
                        │            │
       ┌────────────────▼─┐        ┌─▼────────────────┐
       │ NLLB-200 Engine  │        │  Marian Engine   │
       │  (2460 MB RSS)   │        │   (300 MB RSS)   │
       └────────────────┬─┘        └─┬────────────────┘
                        │            │
                        └─────┬──────┘
                              │
                  ┌───────────▼────────────┐
                  │  DNT Unmasking Engine  │
                  └───────────┬────────────┘
                              │
                  ┌───────────▼────────────┐
                  │  QC Validator & Logger │
                  └────────────────────────┘
```

 ### Key Subsystems

 - **DNT Entity Masking Engine**: Scans input text using regular expressions to encapsulate protected entities (brand names, technical terms, numerical identifiers) into standard tokens (e.g., `<dnt_0>`). Masking occurs prior to tokenization to prevent lexical translation of protected terms.
- **Language Router**: Dynamically selects execution routes based on language pair availability. Primary routes route directly to NLLB-200. Secondary routes leverage Marian models with Least Recently Used (LRU) caching. When direct language pairs are absent in Marian, the router executes a two-stage English pivot (`L_src → en → L_tgt`).
- **Execution & Thread Control**: CPU execution is globally throttled using `torch.set_num_threads(2)` to mitigate thread thrashing, prevent OS core saturation, and cap system memory high-water marks.

---

 ## 2\. Benchmark Evaluation & Engine Trade-Offs

 Quantitative benchmarks were captured across 36 evaluation samples on an Intel CPU runtime environment. Memory peak high-water mark across full model execution was recorded at 2906.18 MB RSS.

 | Metric | Model Family 1 (NLLB-200) | Model Family 2 (Marian) | Unit / Scale |
| --- | --- | --- | --- |
| **Evaluated Samples** | **36** | **23** _(13 skipped - unsupported pairs)_ | Count |
| **BLEU Score** | **53.34** | **40.73** | SacreBLEU scale \[0–100\] |
| **chrF Score** | **75.34** | **64.15** | Character F-score \[0–100\] |
| **DNT Pass Rate** | **88.9%** | **87.0%** | Percentage intact |
| **Avg Latency** | **1868.02** | **5015.87** | ms / segment |
| **Throughput** | **3.46** | **1.34** | tokens / sec |
| **Peak RAM** | 2906.18 | 2906.18 | Megabytes (RSS) |
| **Asset footprint** | \~2460.00 | \~300.00 | Megabytes per model instance |

 ### Performance & Quality Analysis

 - **NLLB-200**: Serves as the high-accuracy primary engine. Achieving a BLEU score of **53.34** and a chrF score of **75.34**, it maintains semantic fidelity across low-resource language pairs. Throughput averages **3.46 tokens/sec** with a single model binary in memory.
- **Marian**: Operates as a lightweight alternative (\~300 MB asset size). While memory-efficient during load time, total execution latency increases to **5015.87 ms/segment** on un-cached multi-step pivot routes (`L_src → en → L_tgt`) due to sequential model loading overhead and dual-pass inference.

---

 ## 3\. Quality Control (QC) & Fault Detection

 The system incorporates a rule-based post-translation evaluation module (`cli.py qc`) designed to intercept translation failures, placeholder corruption, and numerical discrepancies before output generation.

```
                        QC Evaluation Pipeline

   Input JSON (25 samples) ──► Rules Engine ──┬──► Passed (17 / 68.0%)
                                              │
                                              └──► Flagged (8 / 32.0%) ──► review_queue.json
```

 ### QC Validation Performance

 A test set of 25 QC samples containing injected faults (untranslated text, corrupted DNT tokens, mutated numbers) was processed:

 - **Total Processed**: 25 samples
- **Passed Cleanly**: 17 samples (68.0%)
- **Flagged for Review**: 8 samples (32.0%)
- **Precision**: `0.625`
- **Recall**: `1.000`
- **F₁ Score**: `0.7692`
- **Execution Time**: `< 1.00 second`

 A Recall score of **1.000** verifies zero false negatives: every injected critical error was successfully trapped and routed to `data/outputs/review_queue.json` for manual oversight.

---

 ## 4\. Failure Analysis & Mitigations

```
                            System Bottlenecks & Fixes

   Failure Point                 Root Cause                          Engineering Mitigation
┌────────────────────────┐    ┌──────────────────────────┐         ┌───────────────────────────┐
│ Thread Lockup & Freeze │ ──►│ Uncapped PyTorch CPU     │ ───────►│ Capped to 2 threads       │
│ (BENCH-00)             │    │ worker pool (8 threads)  │         │ (torch.set_num_threads)   │
└────────────────────────┘    └──────────────────────────┘         └───────────────────────────┘

┌────────────────────────┐    ┌──────────────────────────┐         ┌───────────────────────────┐
│ DNT Mask Corruption    │ ──►│ Subword tokenizers       │ ───────►│ Regex post-processing     │
│ (DNT Preservation)     │    │ splitting <dnt_0> tags   │         │ & pattern restoration     │
└────────────────────────┘    └──────────────────────────┘         └───────────────────────────┘
```

 ### Identified Failures & Resolutions

 1. **CPU Memory Thrashing (BENCH-00)**
    Initial stress testing under default multi-threading settings (8 CPU worker threads) caused disk thrashing and process lockups due to RAM overhead exceeding physical memory limits.
    **Fix:** Restricted PyTorch thread count to 2 threads (`torch.set_num_threads(2)`) and implemented LRU model caching to limit active memory usage to \~2.9 GB.
2. **Subword Tokenizer Corruption**
    Certain subword tokenizers fragmented placeholder tags (e.g., `<dnt_0>` split into `<`, `dnt`, `_0`), leading to unmasking failures.
    **Fix:** Enhanced regular expression pattern matching to sanitize subword space tokens (` ` / `##`) adjacent to placeholder tags prior to unmasking.

---

 ## 5\. Automated Test Suite Audit

 The full test suite was validated using `pytest` (version 9.1.1) across Python 3.10.0 on Windows CPU architecture.

 - **Total Test Cases**: 37 collected
- **Passed**: 37 (100% pass rate)
- **Total Runtime**: 144.29 seconds (2 min 24 sec)

 ### Test Coverage Breakdown

 | Test Suite | Tests | Pass Rate |
| --- | --- | --- |
| `tests/integration/test_20_non_english_routes.py` | 20 | 100% |
| `tests/integration/test_pipeline_e2e.py` | 1 | 100% |
| `tests/integration/test_target_matrix.py` | 1 | 100% |
| `tests/unit/test_engines.py` | 4 | 100% |
| `tests/unit/test_qc_rules.py` | 9 | 100% |
| `tests/unit/test_router.py` | 2 | 100% |
| **Total** | **37** | **100%** |

---

 ## 6\. Product Roadmap & Future Improvements

```
                            Product Roadmap

 Phase 1 (Current)          Phase 2 (Near-Term)          Phase 3 (Production)
┌──────────────────┐       ┌────────────────────┐       ┌──────────────────────┐
│ • PyTorch CPU    │ ────► │ • ONNX Runtime     │ ────► │ • Async Queue        │
│ • 2-Thread Cap   │       │ • INT8 Quantization│       │ • Fine-tuned LLM QC  │
│ • Rule QC        │       │ • GPU Auto-Fallback│       │ • Distributed Redis  │
└──────────────────┘       └────────────────────┘       └──────────────────────┘
```

 ### 1\. Model Quantization

 Convert NLLB-200 weights to INT8 precision via ONNX Runtime to reduce memory footprint from approximately **2460 MB to \<800 MB** and improve CPU inference throughput by an estimated **2.5×**.

 ### 2\. Asynchronous Batch Execution

 Implement a background task queue (Celery/Redis) to handle pivot-route translation requests asynchronously without blocking thread execution.

 ### 3\. ML-Assisted Quality Estimation

 Upgrade the rule-based QC validator with lightweight Quality Estimation models (e.g., CometKiwi or sentence-transformer embeddings) to flag semantic drift alongside hard rule violations.