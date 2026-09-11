from typing import Dict, Any, Optional
from src.qc.entity_checker import EntityChecker
from src.qc.lid import LanguageIdentifier
from src.qc.anomaly_checker import AnomalyChecker
from src.qc.back_translation import BackTranslationValidator
from src.utils.masker import get_masker
from src.utils.logger import setup_logger

logger = setup_logger()


class QCPipeline:
    """Engine-agnostic Quality Control Pipeline combining independent quality gates."""

    def __init__(self, enable_back_translation: bool = True):
        self.entity_checker = EntityChecker()
        self.lid = LanguageIdentifier()
        self.anomaly_checker = AnomalyChecker()
        self.enable_back_translation = enable_back_translation
        self._back_validator: Optional[BackTranslationValidator] = None

    @property
    def back_validator(self) -> Optional[BackTranslationValidator]:
        if self._back_validator is None and self.enable_back_translation:
            self._back_validator = BackTranslationValidator()
        return self._back_validator

    def evaluate_segment(
        self,
        original_src: str,
        target_lang: str,
        final_translation: Optional[str] = None,
        translated_text: Optional[str] = None,
        translated_masked: Optional[str] = None,
        entity_mapping: Optional[Dict[str, str]] = None,
        source_lang: str = "en",
        forward_engine: Optional[str] = None,
        engine_used: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        mapping = entity_mapping or {}
        output_text = (final_translation or translated_text or "").strip()
        engine = engine_used or forward_engine

        # 1. Anomaly Gate
        anomaly_qc = self.anomaly_checker.check_anomalies(output_text, original_src=original_src)

        # 2. DNT & Entity Preservation Gate
        entity_qc = self.entity_checker.check_preservation(mapping, output_text)

        # 3. Intermediate Raw Masked Placeholder QC Check
        masker_cls = get_masker(engine)
        placeholder_qc = masker_cls.validate(translated_masked or "", mapping) if translated_masked else {"valid": True}

        dnt_qc = {
            "valid": entity_qc.get("passed", True) and placeholder_qc.get("valid", True),
            "missing_entities": entity_qc.get("missing_entities", []),
            "status": "verified_final_output" if placeholder_qc.get("valid", True) else "placeholder_corrupted"
        }

        # 4. Language Identification Gate
        lang_qc = self.lid.verify_language(output_text, target_lang)

        # 5. Back-Translation Quality Gate
        bt_qc = {"passed": True, "bleu_score": None, "chrf_score": None}
        if self.enable_back_translation and self.back_validator and output_text and anomaly_qc["passed"]:
            try:
                independent_engine = "m2m100" if engine == "marian" else "marian"
                bt_qc = self.back_validator.evaluate(
                    original_src_text=original_src,
                    target_translated_text=output_text,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    forced_back_engine=independent_engine
                )
            except Exception as e:
                logger.error(f"QC Back-translation error: {e}")
                bt_qc = {"passed": False, "error": str(e)}

        overall_passed = (
            bool(anomaly_qc.get("passed", False))
            and bool(entity_qc.get("passed", False))
            and bool(placeholder_qc.get("valid", True))
            and bool(lang_qc.get("passed", False))
            and bool(bt_qc.get("passed", True))
        )

        return {
            "overall_passed": overall_passed,
            "anomaly_check": anomaly_qc,
            "dnt_validation": dnt_qc,
            "entity_preservation": entity_qc,
            "language_identification": lang_qc,
            "back_translation": bt_qc
        }