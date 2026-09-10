import hashlib
from typing import Optional
from diskcache import Cache


class TranslationCache:
    """Persistent SQLite-backed cache using diskcache for fast lookup and resumption."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache = Cache(cache_dir)

    def _generate_key(self, text: str, src_lang: str, tgt_lang: str, model_name: str) -> str:
        src_clean = src_lang.lower().strip()
        tgt_clean = tgt_lang.lower().strip()
        raw_key = f"{src_clean}:{tgt_clean}:{model_name}:{text.strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, text: str, src_lang: str, tgt_lang: str, model_name: str) -> Optional[str]:
        key = self._generate_key(text, src_lang, tgt_lang, model_name)
        return self.cache.get(key)

    def set(self, text: str, src_lang: str, tgt_lang: str, model_name: str, translation: str):
        key = self._generate_key(text, src_lang, tgt_lang, model_name)
        self.cache.set(key, translation)

    def clear(self):
        self.cache.clear()

    def close(self):
        self.cache.close()