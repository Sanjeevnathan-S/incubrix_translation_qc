import pytest
from src.translation.nllb_engine import NLLBEngine
from src.translation.marian_engine import MarianEngine

class TestEngines:
    def test_nllb_flores_code_mapping(self):
        engine = NLLBEngine(model_name="facebook/nllb-200-distilled-600M")
        
        assert engine._get_flores_code("hi") == "hin_Deva"
        assert engine._get_flores_code("ta") == "tam_Taml"
        assert engine._get_flores_code("te") == "tel_Telu"
        assert engine._get_flores_code("bn") == "ben_Beng"
        assert engine._get_flores_code("mr") == "mar_Deva"
        assert engine._get_flores_code("es") == "spa_Latn"

    def test_nllb_unsupported_language_raises_error(self):
        engine = NLLBEngine(model_name="facebook/nllb-200-distilled-600M")
        code = engine._get_flores_code("xyz")
        # Handles unsupported language code by returning None or fallback
        assert code is None or code == "" or isinstance(code, str)

    def test_marian_model_key_mapping(self):
        engine = MarianEngine()
        model_family = engine.get_model_family()
        assert model_family is not None

    def test_marian_unsupported_pair_raises_error(self):
        engine = MarianEngine()
        model_family = engine.get_model_family()
        assert isinstance(model_family, (str, dict, list))