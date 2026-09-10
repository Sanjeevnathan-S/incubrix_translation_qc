from typing import Dict, Any, Optional
from src.qc.entity_checker import EntityChecker
from src.qc.lid import LanguageIdentifier
from src.qc.anomaly_checker import AnomalyChecker
from src.qc.back_translation import BackTranslationValidator
from src.utils.logger import setup_logger

logger = setup_logger()


class QCPipeline:
    """Engine-agnostic Quality Control Pipeline combining 4 independent quality gates."""

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
        entity_mapping: Optional[Dict[str, str]] = None,
        source_lang: str = "en",
        forward_engine: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluates translated output against input parameters using independent QC gates.
        """
        mapping = entity_mapping or {}
        output_text = (final_translation or translated_text or "").strip()

        # 1. Anomaly Gate (Repetition loops, Source Copy, Length Ratio)
        anomaly_qc = self.anomaly_checker.check_anomalies(output_text, original_src=original_src)

        # 2. DNT & Entity Preservation Gate
        entity_qc = self.entity_checker.check_preservation(mapping, output_text)
        dnt_qc = {
            "valid": entity_qc.get("passed", True),
            "missing_entities": entity_qc.get("missing_entities", []),
            "status": "verified_final_output"
        }

        # 3. Language Identification Gate (FastText LID)
        lang_qc = self.lid.verify_language(output_text, target_lang)

        # 4. Back-Translation Quality Gate
        bt_qc = {"passed": True, "bleu_score": None, "chrf_score": None}
        if self.enable_back_translation and self.back_validator and output_text and anomaly_qc["passed"]:
            try:
                independent_engine = "m2m100" if forward_engine == "marian" else "marian"
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