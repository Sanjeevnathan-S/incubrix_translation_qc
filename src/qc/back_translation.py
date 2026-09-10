import inspect
from typing import Dict, Any, Optional
from src.routing.router import TranslationRouter
from src.utils.metrics import calculate_bleu, calculate_chrf
from src.utils.logger import setup_logger

logger = setup_logger()


class BackTranslationValidator:
    """Validates translation quality by translating target text back to the source language."""

    def __init__(self, config_path: str = "config/routing_table.yaml", cache_dir: str = "data/cache"):
        self.router = TranslationRouter(config_path=config_path, cache_dir=cache_dir)

    def evaluate(
        self,
        original_src_text: str,
        target_translated_text: str,
        target_lang: str,
        source_lang: str = "en",
        bleu_threshold: float = 20.0,
        forced_back_engine: Optional[str] = "marian"
    ) -> Dict[str, Any]:
        """
        Translates `target_translated_text` back into `source_lang` using an independent model family.
        """
        if not target_translated_text or not target_translated_text.strip():
            return {
                "passed": False,
                "bleu_score": 0.0,
                "chrf_score": 0.0,
                "back_translation": "",
                "error": "empty_input"
            }

        try:
            # Inspect router method signature to safely map the engine parameter across implementation versions
            sig = inspect.signature(self.router.translate_segment)
            kwargs = {
                "text": target_translated_text,
                "src_lang": target_lang,
                "tgt_lang": source_lang,
            }

            if "forced_engine" in sig.parameters:
                kwargs["forced_engine"] = forced_back_engine
            elif "engine" in sig.parameters:
                kwargs["engine"] = forced_back_engine
            elif "preferred_engine" in sig.parameters:
                kwargs["preferred_engine"] = forced_back_engine

            if "use_cache" in sig.parameters:
                kwargs["use_cache"] = True

            router_res = self.router.translate_segment(**kwargs)

            if isinstance(router_res, tuple):
                back_translated = router_res[0]
                engine_used = router_res[1] if len(router_res) > 1 else forced_back_engine
            else:
                back_translated = str(router_res)
                engine_used = forced_back_engine

            bleu = calculate_bleu(back_translated, original_src_text)
            chrf = calculate_chrf(back_translated, original_src_text)

            return {
                "passed": bleu >= bleu_threshold,
                "bleu_score": bleu,
                "chrf_score": chrf,
                "back_translation": back_translated,
                "engine_used": engine_used
            }
        except Exception as e:
            logger.error(f"Back-translation evaluation step failed: {e}")
            return {
                "passed": False,
                "bleu_score": 0.0,
                "chrf_score": 0.0,
                "error": str(e)
            }