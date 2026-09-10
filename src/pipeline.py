import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.utils.loader import load_input_segments
from src.utils.masker import get_masker
from src.utils.logger import setup_logger, get_peak_memory_mb
from src.routing.router import TranslationRouter
from src.qc.pipeline import QCPipeline

logger = setup_logger()


class TranslationPipeline:
    """End-to-end Translation Execution Pipeline with engine-aware DNT masking and integrated QC validation."""

    def __init__(
        self,
        config_path: str = "config/routing_table.yaml",
        cache_dir: str = "data/cache",
        enable_qc_back_translation: bool = True
    ):
        self.router = TranslationRouter(config_path=config_path, cache_dir=cache_dir)
        self.qc_pipeline = QCPipeline(enable_back_translation=enable_qc_back_translation)

    def process_segment(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        dnt_terms: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Processes a segment using dynamic engine selection and engine-specific placeholder masking."""
        # 1. Determine target translation engine from router
        target_engine = self.router.select_engine(src_lang, tgt_lang)
        masker = get_masker(target_engine)

        # 2. Mask entities using engine-specific format ([1] for Marian, __DNT_001__ for others)
        masked_text, entity_mapping = masker.mask(text, dnt_terms=dnt_terms)

        # 3. Translate masked segment via engine
        translated_masked, engine_used, was_cached = self.router.translate_segment(
            text=masked_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            use_cache=use_cache
        )

        # 4. Unmask entities using engine-specific restoration rules
        final_translation = masker.unmask(translated_masked, entity_mapping)

        # 5. Evaluate QC metrics passing engine context for validation
        qc_result = self.qc_pipeline.evaluate_segment(
            original_src=text,
            target_lang=tgt_lang,
            translated_masked=translated_masked,
            final_translation=final_translation,
            entity_mapping=entity_mapping,
            source_lang=src_lang,
            engine_used=engine_used
        )

        return {
            "source_text": text,
            "translated_text": final_translation,
            "engine_used": engine_used,
            "is_cached": was_cached,
            "qc_metrics": qc_result
        }

    def process_file(
        self,
        input_file: str,
        output_file: str,
        src_lang: str,
        tgt_lang: str,
        dnt_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Batch processes an input dataset file and exports translations with QC metadata."""
        start_time = time.time()
        segments = load_input_segments(input_file)
        results = []
        passed_qc_count = 0

        for seg in segments:
            text = seg.get("text", "")
            seg_id = seg.get("id")

            res = self.process_segment(
                text=text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dnt_terms=dnt_terms
            )
            res["id"] = seg_id

            if res["qc_metrics"]["overall_passed"]:
                passed_qc_count += 1

            results.append(res)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        total_time = round(time.time() - start_time, 2)
        total_segs = len(segments)

        summary = {
            "total_segments": total_segs,
            "qc_pass_rate": f"{(passed_qc_count / total_segs) * 100:.1f}%" if total_segs > 0 else "0.0%",
            "total_latency_sec": total_time,
            "latency_per_segment_ms": round((total_time / total_segs) * 1000, 2) if total_segs > 0 else 0,
            "peak_memory_mb": get_peak_memory_mb(),
            "output_file": str(output_path)
        }

        logger.info(f"Batch translation completed: {summary}")
        return summary