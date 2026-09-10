from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Source text to translate", example="Your order #89201 has been dispatched via IncuBrix Express.")
    src_lang: str = Field(..., description="Source language code (ISO 639-1 / NLLB format)", example="en")
    tgt_lang: str = Field(..., description="Target language code (ISO 639-1 / NLLB format)", example="hi")
    dnt_terms: Optional[List[str]] = Field(default_factory=list, description="Explicit Do-Not-Translate terms", example=["IncuBrix Express"])
    use_cache: bool = Field(True, description="Enable translation caching")


class TranslationResponse(BaseModel):
    id: Optional[str] = None
    source_text: str
    translated_text: str
    engine_used: str
    is_cached: bool
    latency_ms: float
    qc_metrics: Dict[str, Any]


class BatchTranslationRequest(BaseModel):
    segments: List[TranslationRequest]


class BatchTranslationResponse(BaseModel):
    total_segments: int
    total_latency_ms: float
    results: List[TranslationResponse]