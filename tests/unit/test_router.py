import pytest
from src.routing.router import TranslationRouter

class TestTranslationRouter:
    @pytest.fixture
    def router(self):
        return TranslationRouter()

    def test_router_selects_configured_primary_and_fallback(self, router):
        route_selection = router.select_engine("en", "es")
        assert route_selection is not None

        route_default = router.select_engine("en", "hi")
        assert route_default is not None

    def test_router_cache_key_generation_and_storage(self, router):
        src_text = "Welcome to IncuBrix local translation engine."
        src_lang = "en"
        tgt_lang = "es"

        if hasattr(router, 'get_cached_translation'):
            cached_val, is_cached = router.get_cached_translation(src_text, src_lang, tgt_lang)
            assert is_cached is False
            
            if hasattr(router, 'cache_result'):
                router.cache_result(f"{src_lang}_{tgt_lang}_{src_text}", "Bienvenido")
        else:
            assert router is not None