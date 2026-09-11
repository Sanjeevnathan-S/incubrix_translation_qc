import time
from typing import List
from fastapi import FastAPI, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware


from src.api.schemas import (
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse,
)
from src.pipeline import TranslationPipeline
from src.utils.logger import setup_logger

logger = setup_logger()

app = FastAPI(
    title="Multilingual Translation API",
    version="1.0.0",
    description="Track 04 Pipeline with Multi-tier Automated Quality Control & DNT Term Preservation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = TranslationPipeline(
    config_path="config/routing_table.yaml",
    cache_dir="data/cache",
    enable_qc_back_translation=False
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "multilingual-translation-api"}


@app.post("/v1/translate", response_model=TranslationResponse, status_code=status.HTTP_200_OK)
def translate_single(
    request: TranslationRequest = Body(
        ...,
        openapi_examples={
            "qc_15_en_hi": {
                "summary": "qc_15_en_hi (Paytm DNT - Passed)",
                "description": "Verified passing test case from dataset. DNT term: Paytm.",
                "value": {
                    "text": "Please pay using Paytm to receive cashback.",
                    "src_lang": "en",
                    "tgt_lang": "hi",
                    "dnt_terms": ["Paytm"],
                    "use_cache": True
                }
            },
            "qc_16_en_es": {
                "summary": "qc_16_en_es (UltraDrive DNT - Passed)",
                "description": "Verified passing test case from dataset. DNT term: UltraDrive.",
                "value": {
                    "text": "Save all your project files directly to UltraDrive.",
                    "src_lang": "en",
                    "tgt_lang": "es",
                    "dnt_terms": ["UltraDrive"],
                    "use_cache": True
                }
            },
            "qc_05_ta_fr": {
                "summary": "qc_05_ta_fr (PassExpress DNT - Passed)",
                "description": "Verified passing test case from dataset. DNT term: PassExpress.",
                "value": {
                    "text": "PassExpress வழியாக உங்கள் டிக்கெட்டைப் பெறுங்கள்.",
                    "src_lang": "ta",
                    "tgt_lang": "fr",
                    "dnt_terms": ["PassExpress"],
                    "use_cache": True
                }
            },
            "qc_22_en_ta": {
                "summary": "qc_22_en_ta (IncuBrix DNT - Passed)",
                "description": "Verified passing test case from dataset. DNT term: IncuBrix.",
                "value": {
                    "text": "Welcome to the IncuBrix portal.",
                    "src_lang": "en",
                    "tgt_lang": "ta",
                    "dnt_terms": ["IncuBrix"],
                    "use_cache": True
                }
            }
        }
    )
):
    """
    Translates a single text segment while ensuring DNT term preservation.
    """
    start_time = time.perf_counter()
    try:
        res = pipeline.process_segment(
            text=request.text,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
            dnt_terms=request.dnt_terms,
            use_cache=request.use_cache
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return TranslationResponse(
            id=res.get("id"),
            source_text=res["source_text"],
            translated_text=res["translated_text"],
            engine_used=res["engine_used"],
            is_cached=res["is_cached"],
            latency_ms=elapsed_ms,
            qc_metrics=res["qc_metrics"]
        )
    except Exception as e:
        logger.error(f"API Translation Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation processing failed: {str(e)}"
        )


@app.post("/v1/translate/batch", response_model=BatchTranslationResponse, status_code=status.HTTP_200_OK)
def translate_batch(
    request: BatchTranslationRequest = Body(
        ...,
        openapi_examples={
            "dnt_passed_dataset_suite": {
                "summary": "Verified DNT Passed Suite",
                "description": "Batch request composed exclusively of verified passing DNT samples (qc_15, qc_16, qc_18, qc_22).",
                "value": {
                    "segments": [
                        {
                            "text": "Please pay using Paytm to receive cashback.",
                            "src_lang": "en",
                            "tgt_lang": "hi",
                            "dnt_terms": ["Paytm"],
                            "use_cache": True
                        },
                        {
                            "text": "Save all your project files directly to UltraDrive.",
                            "src_lang": "en",
                            "tgt_lang": "es",
                            "dnt_terms": ["UltraDrive"],
                            "use_cache": True
                        },
                        {
                            "text": "Pesanan Anda di Tokopedia telah dikirim.",
                            "src_lang": "id",
                            "tgt_lang": "en",
                            "dnt_terms": ["Tokopedia"],
                            "use_cache": True
                        },
                        {
                            "text": "Welcome to the IncuBrix portal.",
                            "src_lang": "en",
                            "tgt_lang": "ta",
                            "dnt_terms": ["IncuBrix"],
                            "use_cache": True
                        }
                    ]
                }
            }
        }
    )
):
    """
    Processes a batch of translation segments with verified DNT preservation.
    """
    start_time = time.perf_counter()
    results: List[TranslationResponse] = []

    for item in request.segments:
        seg_start = time.perf_counter()
        try:
            res = pipeline.process_segment(
                text=item.text,
                src_lang=item.src_lang,
                tgt_lang=item.tgt_lang,
                dnt_terms=item.dnt_terms,
                use_cache=item.use_cache
            )
            seg_latency = round((time.perf_counter() - seg_start) * 1000, 2)

            results.append(
                TranslationResponse(
                    id=res.get("id"),
                    source_text=res["source_text"],
                    translated_text=res["translated_text"],
                    engine_used=res["engine_used"],
                    is_cached=res["is_cached"],
                    latency_ms=seg_latency,
                    qc_metrics=res["qc_metrics"]
                )
            )
        except Exception as e:
            logger.error(f"Batch Item Processing Exception: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch item translation failed: {str(e)}"
            )

    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return BatchTranslationResponse(
        total_segments=len(results),
        total_latency_ms=total_latency_ms,
        results=results
    )

