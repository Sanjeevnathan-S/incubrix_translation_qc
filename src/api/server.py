import time
from typing import List
from fastapi import FastAPI, HTTPException, status
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
    description="Track 04 Pipeline with Multi-tier Automated Quality Control"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global translation pipeline state
pipeline = TranslationPipeline(
    config_path="config/routing_table.yaml",
    cache_dir="data/cache",
    enable_qc_back_translation=False
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "multilingual-translation-api"}


@app.post("/v1/translate", response_model=TranslationResponse, status_code=status.HTTP_200_OK)
def translate_single(request: TranslationRequest):
    """
    Translates a single text segment.
    Handles entity detection, engine-aware masking ([1] vs __DNT_001__),
    dynamic routing (Marian vs NLLB), translation, unmasking, and QC.
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
def translate_batch(request: BatchTranslationRequest):
    """
    Processes a batch of translation segments sequentially.
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
                    source_text=res["source_text"],
                    translated_text=res["translated_text"],
                    engine_used=res["engine_used"],
                    is_cached=res["is_cached"],
                    latency_ms=seg_latency,
                    qc_metrics=res["qc_metrics"]
                )
            )
        except Exception as e:
            logger.error(f"Batch Item Error: {str(e)}")
            results.append(
                TranslationResponse(
                    source_text=item.text,
                    translated_text=item.text,
                    engine_used="fallback",
                    is_cached=False,
                    latency_ms=0.0,
                    qc_metrics={"overall_passed": False, "error": str(e)}
                )
            )

    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return BatchTranslationResponse(
        total_segments=len(results),
        total_latency_ms=total_latency_ms,
        results=results
    )