from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTranslationEngine(ABC):
    """Abstract interface defining standard contract across model families."""

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translates a single string segment."""
        pass

    @abstractmethod
    def batch_translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Translates a batch of text segments."""
        pass

    @abstractmethod
    def get_model_family(self) -> str:
        """Returns the distinct model family identifier for independent QC routing."""
        pass