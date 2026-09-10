import pytest
from src.routing.router import TranslationRouter

NON_ENGLISH_20_DIRECTIONS = [
    ("hi", "ta"), ("ta", "bn"), ("bn", "mr"), ("es", "fr"), ("fr", "de"),
    ("de", "pt"), ("pt", "id"), ("id", "hi"), ("te", "bn"), ("mr", "es"),
    ("de", "ta"), ("es", "pt"), ("fr", "id"), ("hi", "de"), ("ta", "es"),
    ("bn", "fr"), ("mr", "pt"), ("te", "id"), ("id", "de"), ("es", "hi")
]


class TestNonEnglishRoutingMatrix:
    @pytest.fixture
    def router(self, tmp_path):
        return TranslationRouter(cache_dir=str(tmp_path / "cache"))

    @pytest.mark.parametrize("src,tgt", NON_ENGLISH_20_DIRECTIONS)
    def test_non_english_route_resolution(self, router, src, tgt):
        engine = router.select_engine(src, tgt)
        assert isinstance(engine, str)
        assert engine in ["nllb", "marian"]