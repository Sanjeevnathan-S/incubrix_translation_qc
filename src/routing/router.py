import os
from typing import Dict, Any, Tuple, List, Optional
import yaml
from loguru import logger

from src.translation.base import BaseTranslationEngine
from src.translation.nllb_engine import NLLBEngine
from src.translation.marian_engine import MarianEngine
from src.routing.cache import TranslationCache


class TranslationRouter:
    """
    Orchestrates translation engines based on config, manages cached hits,
    selects engines dynamically based on entity presence and Marian model support,
    and executes automatic fallbacks on engine failure.
    """

    def __init__(self, config_path: str = "config/routing_table.yaml", cache_dir: str = "data/cache"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.cache = TranslationCache(cache_dir)
        self._engines: Dict[str, BaseTranslationEngine] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            logger.warning(f"Config path {config_path} not found. Loading defaults.")
            return {"default_engine": "nllb", "fallback_engine": "marian", "routes": {}}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _get_engine_instance(self, engine_key: str) -> BaseTranslationEngine:
        if engine_key not in self._engines:
            if engine_key == "nllb":
                self._engines[engine_key] = NLLBEngine()
            elif engine_key == "marian":
                self._engines[engine_key] = MarianEngine()
            else:
                raise ValueError(f"Unknown or unsupported translation engine: '{engine_key}'")
        return self._engines[engine_key]

    def get_engine(self, engine_key: str) -> BaseTranslationEngine:
        """Exposes engine instances to external pipeline components."""
        return self._get_engine_instance(engine_key)

    def is_marian_supported(self, src_lang: str, tgt_lang: str) -> bool:
        """Checks if a language pair exists in Marian's PAIR_CONFIG dictionary."""
        src_clean = MarianEngine.LANG_MAP.get(src_lang.lower().strip(), src_lang.lower().strip())
        tgt_clean = MarianEngine.LANG_MAP.get(tgt_lang.lower().strip(), tgt_lang.lower().strip())
        return (src_clean, tgt_clean) in MarianEngine.PAIR_CONFIG

    def is_nllb_supported(self, src_lang: str, tgt_lang: str) -> bool:
        """Checks if both source and target languages exist in NLLB's LANG_MAP."""
        src_clean = src_lang.lower().strip()
        tgt_clean = tgt_lang.lower().strip()
        nllb_langs = set(NLLBEngine.LANG_MAP.keys()).union(
            {v.lower() for v in NLLBEngine.LANG_MAP.values()}
        )
        return src_clean in nllb_langs and tgt_clean in nllb_langs

    def _resolve_route(self, src_lang: str, tgt_lang: str) -> Tuple[str, Optional[str]]:
        pair_key = f"{src_lang.lower().strip()}-{tgt_lang.lower().strip()}"
        routes = self.config.get("routes", {})
        route_info = routes.get(pair_key, {})

        primary = route_info.get("primary") if "primary" in route_info else self.config.get("default_engine", "nllb")
        fallback = route_info.get("fallback") if "fallback" in route_info else self.config.get("fallback_engine", "marian")

        return primary, fallback

    def select_engine(
        self, src_lang: str, tgt_lang: str, has_dnt: bool = False
    ) -> str:
        """
        Determines the optimal translation engine ('nllb' vs 'marian').
        - If text HAS DNT terms:
            1. Prefers NLLB if supported for optimal entity placeholder preservation.
            2. If NOT supported by NLLB but IS supported by Marian, routes to Marian.
            3. Otherwise defaults to NLLB.
        - If text has NO DNT terms:
            1. Uses Marian if configured as primary and supported.
            2. Defaults to NLLB if supported, or Marian if unsupported by NLLB.
        """
        src_clean = src_lang.lower().strip()
        tgt_clean = tgt_lang.lower().strip()
        primary_name, fallback_name = self._resolve_route(src_clean, tgt_clean)

        marian_supported = self.is_marian_supported(src_clean, tgt_clean)
        nllb_supported = self.is_nllb_supported(src_clean, tgt_clean)

        # Rule 1: DNT presence routing
        if has_dnt:
            if nllb_supported:
                return "nllb"
            # Scenario: Has DNT, unsupported by NLLB, but supported by Marian
            if marian_supported:
                return "marian"
            return "nllb"

        # Rule 2: Clean text routing (No DNT)
        if primary_name == "marian" and marian_supported:
            return "marian"

        if nllb_supported:
            return "nllb"

        return "marian" if marian_supported else "nllb"

    def translate_segment_with_engine(
        self, engine_name: str, text: str, src_lang: str, tgt_lang: str, use_cache: bool = True
    ) -> Tuple[str, str, bool]:
        """Translates text explicitly using the specified engine name."""
        if not text.strip():
            return "", engine_name, False

        src_clean = src_lang.lower().strip()
        tgt_clean = tgt_lang.lower().strip()

        if use_cache:
            cached_res = self.cache.get(text, src_clean, tgt_clean, engine_name)
            if cached_res is not None:
                return cached_res, engine_name, True

        engine = self._get_engine_instance(engine_name)
        result = engine.translate(text, src_clean, tgt_clean)

        if use_cache:
            self.cache.set(text, src_clean, tgt_clean, engine_name, result)

        return result, engine_name, False

    def translate_segment(
        self, text: str, src_lang: str, tgt_lang: str, use_cache: bool = True
    ) -> Tuple[str, str, bool]:
        if not text.strip():
            return "", "none", False

        src_clean = src_lang.lower().strip()
        tgt_clean = tgt_lang.lower().strip()
        primary_name, fallback_name = self._resolve_route(src_clean, tgt_clean)

        # 1. Check persistent cache
        if use_cache:
            cached_res = self.cache.get(text, src_clean, tgt_clean, primary_name)
            if cached_res is not None:
                return cached_res, primary_name, True

        # 2. Primary Engine Execution
        try:
            engine = self._get_engine_instance(primary_name)
            result = engine.translate(text, src_clean, tgt_clean)
            if use_cache:
                self.cache.set(text, src_clean, tgt_clean, primary_name, result)
            return result, primary_name, False
        except Exception as primary_error:
            logger.warning(
                f"Primary engine '{primary_name}' failed for '{src_clean}->{tgt_clean}': {primary_error}"
            )

        # 3. Fallback Execution (if fallback engine exists)
        if fallback_name and fallback_name != primary_name:
            try:
                logger.info(f"Executing fallback engine '{fallback_name}' for '{src_clean}->{tgt_clean}'")
                engine = self._get_engine_instance(fallback_name)
                result = engine.translate(text, src_clean, tgt_clean)
                if use_cache:
                    self.cache.set(text, src_clean, tgt_clean, fallback_name, result)
                return result, fallback_name, False
            except Exception as fallback_error:
                logger.error(f"Fallback engine '{fallback_name}' also failed: {fallback_error}")
                raise RuntimeError(
                    f"Both primary ('{primary_name}') and fallback ('{fallback_name}') engines failed."
                ) from fallback_error

        raise RuntimeError(f"Primary engine '{primary_name}' failed and no fallback engine was defined for route.")

    def translate_batch(
        self, texts: List[str], src_lang: str, tgt_lang: str, use_cache: bool = True
    ) -> Tuple[List[str], str, bool]:
        if not texts:
            return [], "none", False

        src_clean = src_lang.lower().strip()
        tgt_clean = tgt_lang.lower().strip()
        primary_name, fallback_name = self._resolve_route(src_clean, tgt_clean)

        # Process cache lookup for batch
        uncached_indices = []
        results = [None] * len(texts)
        cached_count = 0

        if use_cache:
            for idx, text in enumerate(texts):
                if not text.strip():
                    results[idx] = ""
                    continue
                cached_val = self.cache.get(text, src_clean, tgt_clean, primary_name)
                if cached_val is not None:
                    results[idx] = cached_val
                    cached_count += 1
                else:
                    uncached_indices.append(idx)
        else:
            uncached_indices = list(range(len(texts)))

        if not uncached_indices:
            return results, primary_name, True

        uncached_texts = [texts[i] for i in uncached_indices]

        # Execute translation on uncached texts
        try:
            engine = self._get_engine_instance(primary_name)
            translated = engine.batch_translate(uncached_texts, src_clean, tgt_clean)
            engine_used = primary_name
        except Exception as e:
            logger.warning(f"Primary batch translation failed on '{primary_name}': {e}")
            if fallback_name and fallback_name != primary_name:
                engine = self._get_engine_instance(fallback_name)
                translated = engine.batch_translate(uncached_texts, src_clean, tgt_clean)
                engine_used = fallback_name
            else:
                raise e

        # Write results to output array and cache
        for idx, trans in zip(uncached_indices, translated):
            results[idx] = trans
            if use_cache and texts[idx].strip():
                self.cache.set(texts[idx], src_clean, tgt_clean, engine_used, trans)

        return results, engine_used, cached_count == len(texts)