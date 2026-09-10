import os
import urllib.request
import fasttext
from loguru import logger


class LanguageIdentifier:
    """FastText-based Language Identification validator with confidence thresholding and short-text handling."""

    MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    MODEL_PATH = "data/models/lid.176.ftz"

    LANG_MAP = {
        # ISO-639-3 / FLORES-200 code mappings to FastText 2-letter codes
        "hin": "hi", "tam": "ta", "tel": "te", "ben": "bn", "mar": "mr",
        "spa": "es", "fra": "fr", "deu": "de", "por": "pt", "ind": "id", "eng": "en",
        # Standard ISO-639-1 mappings
        "hi": "hi", "ta": "ta", "te": "te", "bn": "bn", "mr": "mr",
        "es": "es", "fr": "fr", "de": "de", "pt": "pt", "id": "id", "en": "en"
    }

    def __init__(self, model_path: str = None):
        self.model_path = model_path or self.MODEL_PATH
        self._ensure_model_exists()
        fasttext.FastText.eprint = lambda x: None
        self.model = fasttext.load_model(self.model_path)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(f"Downloading FastText LangID model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)

    def verify_language(self, text: str, expected_lang: str, min_confidence: float = 0.35) -> dict:
        if not text or not text.strip():
            return {"passed": False, "detected_lang": "none", "confidence": 0.0, "reason": "empty_input"}

        words = text.split()
        # FastText LID is unreliable on inputs under 3 words; bypass for short phrases
        if len(words) < 3:
            return {
                "passed": True,
                "detected_lang": "short_text_bypassed",
                "expected_lang": expected_lang,
                "confidence": 1.0,
                "reason": "short_text"
            }

        predictions = self.model.predict(text.replace("\n", " "), k=1)
        raw_label = predictions[0][0]
        confidence = float(predictions[1][0])
        detected_lang = raw_label.replace("__label__", "").split("_")[0].lower()

        clean_expected = expected_lang.split("_")[0].split("-")[0].lower()
        mapped_expected = self.LANG_MAP.get(clean_expected, clean_expected)

        lang_match = (detected_lang == mapped_expected) or (detected_lang == clean_expected)
        passed = lang_match and (confidence >= min_confidence)

        reason = "none"
        if not passed:
            reason = "low_confidence" if lang_match else "language_mismatch"

        return {
            "passed": passed,
            "detected_lang": detected_lang,
            "expected_lang": mapped_expected,
            "confidence": round(confidence, 4),
            "reason": reason
        }