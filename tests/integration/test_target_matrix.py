import json
import pytest
from pathlib import Path
from src.routing.router import TranslationRouter
from src.qc.anomaly_checker import AnomalyChecker

DATA_MATRIX_PATH = Path("data/test_matrix_50.json")


class TestTargetLanguageMatrixExecution:
    @pytest.fixture
    def router(self, tmp_path):
        return TranslationRouter(cache_dir=str(tmp_path / "cache"))

    @pytest.fixture
    def anomaly_checker(self):
        return AnomalyChecker()

    def test_50_case_target_matrix_structure_and_routing(self, router, anomaly_checker):
        assert DATA_MATRIX_PATH.exists(), f"Missing matrix dataset at {DATA_MATRIX_PATH}"
        matrix = json.loads(DATA_MATRIX_PATH.read_text(encoding="utf-8"))

        expected_languages = ["hi", "ta", "te", "bn", "mr", "es", "fr", "de", "pt", "id"]
        assert set(matrix.keys()) == set(expected_languages)

        for tgt_lang, samples in matrix.items():
            assert len(samples) >= 5, f"Target language {tgt_lang} requires at least 5 test cases"

            for sample in samples:
                assert "type" in sample
                assert "text" in sample

                qc_res = anomaly_checker.check_anomalies(sample["text"])
                assert qc_res["reason"] != "empty_output"

                engine = router.select_engine("en", tgt_lang)
                assert engine in ["nllb", "marian"]