import pytest
from unittest.mock import MagicMock
from src.pipeline import TranslationPipeline


class TestPipelineEndToEnd:
    @pytest.fixture
    def pipeline(self, tmp_path, monkeypatch):
        pipe = TranslationPipeline(cache_dir=str(tmp_path / "cache"), enable_qc_back_translation=False)
        
        mock_engine = MagicMock()
        mock_engine.translate.return_value = "Bienvenido a IncuBrix"
        monkeypatch.setattr(pipe.router, "_get_engine_instance", lambda key: mock_engine)
        return pipe

    def test_full_segment_processing_lifecycle(self, pipeline):
        res = pipeline.process_segment(
            text="Welcome to IncuBrix",
            src_lang="en",
            tgt_lang="es",
            dnt_terms=["IncuBrix"],
            use_cache=False
        )

        assert isinstance(res, dict)
        assert res["source_text"] == "Welcome to IncuBrix"
        assert "translated_text" in res
        assert "engine_used" in res
        assert "is_cached" in res
        assert "qc_metrics" in res
        assert isinstance(res["qc_metrics"], dict)
        assert "overall_passed" in res["qc_metrics"]