# Track 04: Multilingual Translation with Independent Quality Control

 A Python translation pipeline built with NLLB and MarianMT models, featuring entity preservation for Do-Not-Translate (DNT) terms, persistent response caching, and automated Quality Control (QC) checks.

---

 ## 1\. Clone Instructions

```
# Clone the repository
git clone https://github.com/Sanjeevnathan-S/incubrix_translation_qc.git
cd incubrix_translation_qc

# Create and activate virtual environment
python -m venv venv

# On Linux/macOS or Git Bash:
source venv/bin/activate
# On Windows PowerShell:
# .\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

 ## 2\. Project Structure

```
incubrix_translation_qc/
├── config/
│   └── routing_table.yaml       # Primary & fallback engine route definitions
├── data/
│   ├── cache/                   # Local persistent translation key-value cache
│   ├── outputs/                 # Generated outputs for benchmarks and QC runs
│   └── test_matrix_50.json      # Benchmark test matrix (50 cases across 10 languages)
├── scripts/
│   ├── benchmark.py             # Comparative benchmarking runner
│   ├── download_models.py       # Local model pre-download script
│   └── qc.py                    # QC validation gate workflow script
├── src/
│   ├── api/
│   │   └── server.py            # FastAPI application entrypoint
│   ├── cli.py                   # Typer CLI application entrypoint
│   ├── pipeline.py              # Main end-to-end TranslationPipeline execution manager
│   ├── qc/
│   │   ├── anomaly_checker.py   # Repetition, source copy, and length ratio rules
│   │   ├── entity_checker.py    # DNT term & entity preservation validation
│   │   └── pipeline.py          # Combined QC pipeline evaluator
│   ├── routing/
│   │   ├── cache.py             # Persistent translation cache manager
│   │   └── router.py            # Dynamic engine routing & fallback logic
│   ├── translation/
│   │   ├── base.py              # Abstract base engine class
│   │   ├── marian_engine.py     # MarianMT engine driver
│   │   └── nllb_engine.py       # NLLB-200 engine driver
│   └── utils/
│       ├── loader.py            # Dataset file loaders
│       ├── logger.py            # Benchmarking, latency, and peak RAM logging
│       ├── masker.py            # Engine-specific entity placeholder maskers
│       └── metrics.py           # Evaluation & performance metric calculators
├── tests/
│   ├── integration/             # E2E pipeline and 20 non-English route tests
│   └── unit/                    # Anomaly, entity preservation, and router unit tests
├── AI_USE.md                    # Disclosure of AI assistance
├── SOURCES.md                   # Model attributions and open-source licenses
└── README.md                    # System documentation
```

---

 ## 3\. Models Used, Routing and Caching Strategy

 ### Models Used

 - **Primary Distilled Multilingual Engine:** `facebook/nllb-200-distilled-600M`
- **Lightweight Language-Pair Engine:** `Helsinki-NLP/opus-mt-*` (MarianMT family models)
- **Language Identification:** `FastText LID` for non-production model quality verification.

 ### Routing Strategy (`TranslationRouter`)

 1. **DNT & Placeholder Rule:** If text contains Do-Not-Translate terms, URLs, or entity masks (`has_dnt=True`), routing prioritizes **NLLB** due to placeholder survival rates.
2. **Clean Text Rule:** If no DNT terms exist and MarianMT supports the specific language pair, **MarianMT** is selected for lower latency on CPU.
3. **Fallback Execution:** If the primary engine raises a runtime exception, the pipeline automatically catches the error and executes the target route's secondary fallback engine before raising an error.

 ### Caching Strategy (`TranslationCache`)

 - **Key Hashing:** Computes a SHA-256 key based on `(source_text, src_lang, tgt_lang, engine_name)`.
- **Lookup & Storage:** Checks `data/cache/` before executing model inference. On a hit, returns the pre-translated output with `"is_cached": true`, bypassing model execution.

---

 ## 4\. Model Routing Architecture Diagram

```
                             +-----------------------+
                             |  Input Text Segment   |
                             +-----------------------+
                                         |
                                         v
                         +-------------------------------+
                         | SHA-256 Cache Lookup          |
                         | (data/cache)                  |
                         +-------------------------------+
                            /                         \
                   [Hit]   /                           \  [Miss]
                          v                             v
           +-----------------------------+   +-----------------------------+
           | Return Cached Result        |   | Entity Masking Layer        |
           | ("is_cached": true)         |   | (DNT terms, URLs, Numbers)  |
           +-----------------------------+   +-----------------------------+
                                                        |
                                                        v
                                         +-------------------------------+
                                         | Dynamic Route Evaluator       |
                                         | Has DNT OR Unsupported Marian?|
                                         +-------------------------------+
                                            /                         \
                                  [Yes]    /                           \  [No]
                                          v                             v
                           +-----------------------------+   +-----------------------------+
                           | Primary Engine: NLLB-200    |   | Primary Engine: MarianMT    |
                           +-----------------------------+   +-----------------------------+
                                          \                             /
                                           \                           /
                                            v                         v
                                         +-------------------------------+
                                         | Execution Status Check        |
                                         +-------------------------------+
                                            /                         \
                                 [Success] /                           \  [Exception]
                                          v                             v
                                         |                   +-----------------------------+
                                         |                   | Secondary Fallback Engine   |
                                         |                   +-----------------------------+
                                         |                                  |
                                         v                                  v
                                   +----------------------------------------------+
                                   | Independent Quality Control Pipeline         |
                                   |  - AnomalyChecker (loops, ratios, source copy)|
                                   |  - EntityChecker (DNT preservation)          |
                                   |  - FastText LID / Back-Translation           |
                                   +----------------------------------------------+
                                                        |
                                                        v
                                   +----------------------------------------------+
                                   | Unmask Placeholders & Log Metrics            |
                                   +----------------------------------------------+
```

---

 ## 5\. CLI Run Instructions

 The command-line interface is built using `Typer` and supports model initialization, translation, batch processing, benchmarking, and quality control validation gates.

 ### 1\. Pre-Download Models

 Download and cache all translation engines and FastText models locally before running offline executions:

```
python src/cli.py download
```

 ### 2\. Single Segment Translation (`translate`)

 Translate a single string segment with online entity masking and quality control checks:

```
python src/cli.py translate \
  --text "Welcome to IncuBrix. Visit https://incubrix.com for details." \
  --src en \
  --tgt es \
  --dnt "IncuBrix"
```

 - **Options:**
- `--text` / `-tx`: Source text segment (Required).
- `--src` / `-s`: Source ISO language code (Default: `en`).
- `--tgt` / `-t`: Target ISO language code (Required).
- `--dnt`: Comma-separated Do-Not-Translate terms (Optional).

 ### 3\. Batch File Processing (`batch`)

 Process a JSON or JSONL file through translation and quality control:

```
python src/cli.py batch \
  --input data/test_matrix_50.json \
  --output data/outputs/batch_results.json \
  --src en \
  --tgt es \
  --dnt "IncuBrix"
```

 - **Options:**
- `--input` / `-i`: Input dataset file path (`.json` or `.jsonl`) (Required).
- `--output` / `-o`: Output file path for results (Required).
- `--src` / `-s`: Source ISO language code (Default: `en`).
- `--tgt` / `-t`: Target ISO language code (Required).
- `--dnt`: Comma-separated Do-Not-Translate terms (Optional).

 ### 4\. Comparative Benchmarking (`benchmark`)

 Run comparative latency, throughput, and quality benchmarking across engine families:

```
python src/cli.py benchmark \
  --input data/test_matrix_50.json \
  --output data/outputs/benchmark_results.json
```

 - **Options:**
- `--input` / `-i`: Custom JSON/JSONL benchmark dataset path (Optional).
- `--output` / `-o`: Output results path (Default: `data/outputs/benchmark_results.json`).

 ### 5\. Quality Control Workflow (`qc`)

 Run Quality Control validation gates (Entity preservation, FastText LID, and Back-Translation):

```
python src/cli.py qc \
  --input data/test_matrix_50.json \
  --output data/outputs/qc_results.json
```

 - **Options:**
- `--input` / `-i`: Custom JSON QC input dataset (Optional).
- `--output` / `-o`: Output results path (Default: `data/outputs/qc_results.json`).

---

 ## 6\. Start FASTAPI Server

 Start the local FastAPI web server:

```
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

 Interactive API documentation will be accessible at `http://localhost:8000/docs`.

---

 ## 7\. Pytest Execution

 Run all unit and integration test suites:

```
pytest
```

 Run tests with test coverage reporting:

```
pytest --cov=src --cov-report=term-missing
```