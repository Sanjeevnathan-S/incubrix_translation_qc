import pytest
from src.qc.anomaly_checker import AnomalyChecker
from src.qc.entity_checker import EntityChecker


class TestQCRules:
    @pytest.fixture
    def anomaly_checker(self):
        return AnomalyChecker()

    @pytest.fixture
    def entity_checker(self):
        return EntityChecker()

    def test_anomaly_empty_input(self, anomaly_checker):
        res = anomaly_checker.check_anomalies("")
        assert res["passed"] is False
        assert res["reason"] == "empty_output"

    def test_anomaly_untranslated_source_copy(self, anomaly_checker):
        src = "This is a sentence with more than three words"
        res = anomaly_checker.check_anomalies(src, original_src=src)
        assert res["passed"] is False
        assert res["reason"] == "untranslated_source_copy"

    def test_anomaly_extreme_length_ratio(self, anomaly_checker):
        src = "One two three four five six seven eight nine ten"
        tgt = "One"
        res = anomaly_checker.check_anomalies(tgt, original_src=src)
        assert res["passed"] is False
        assert res["reason"].startswith("extreme_length_ratio_")

    def test_anomaly_consecutive_word_loop(self, anomaly_checker):
        tgt = "this is word word word word repeat"
        res = anomaly_checker.check_anomalies(tgt)
        assert res["passed"] is False
        assert res["reason"] == "consecutive_word_loop"

    def test_anomaly_phrase_repetition_loop(self, anomaly_checker):
        tgt = "hello world hello world hello world hello world"
        res = anomaly_checker.check_anomalies(tgt)
        assert res["passed"] is False
        assert res["reason"] == "phrase_repetition_loop"

    def test_anomaly_valid_translation(self, anomaly_checker):
        src = "Hello my friend"
        tgt = "Hola mi amigo"
        res = anomaly_checker.check_anomalies(tgt, original_src=src)
        assert res["passed"] is True
        assert res["reason"] == "none"

    def test_entity_preservation_empty_mapping(self, entity_checker):
        res = entity_checker.check_preservation({}, "Bienvenido a IncuBrix")
        assert res["passed"] is True
        assert res["score"] == 1.0
        assert res["missing_entities"] == []
        assert res["total_entities"] == 0

    def test_entity_preservation_success(self, entity_checker):
        mapping = {"__DNT_001__": "IncuBrix", "__DNT_002__": "https://incubrix.com"}
        tgt = "Bienvenido a IncuBrix, visita https://incubrix.com para más."
        res = entity_checker.check_preservation(mapping, tgt)
        assert res["passed"] is True
        assert res["score"] == 1.0
        assert len(res["missing_entities"]) == 0

    def test_entity_preservation_missing_entity(self, entity_checker):
        mapping = {"__DNT_001__": "IncuBrix", "__DNT_002__": "https://incubrix.com"}
        tgt = "Bienvenido a IncuBrix sin enlace."
        res = entity_checker.check_preservation(mapping, tgt)
        assert res["passed"] is False
        assert res["score"] == 0.5
        assert len(res["missing_entities"]) == 1
        assert res["missing_entities"][0]["placeholder"] == "__DNT_002__"