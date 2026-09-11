import os
import gc
import re
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.translation.base import BaseTranslationEngine


class MarianEngine(BaseTranslationEngine):
    """
    Secondary Multilingual Translation Engine using MarianMT / OPUS-MT Architecture.
    Handles direct pairs and automatic English pivoting for non-English directions.
    """

    MAX_CACHED_MODELS: int = 3  # Keeps up to 3 Marian models (~900MB total) in RAM

    LANG_MAP: Dict[str, str] = {
        "eng_latn": "en", "hin_deva": "hi", "tam_taml": "ta", "tam_tamx": "ta",
        "tel_telu": "te", "ben_beng": "bn", "mar_deva": "mr", "guj_gujr": "gu",
        "kan_knda": "kn", "mal_mlym": "ml", "pan_guru": "pa", "urd_arab": "ur",
        "spa_latn": "es", "fra_latn": "fr", "deu_latn": "de", "por_latn": "pt",
        "ind_latn": "id",
    }

    PAIR_CONFIG: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {
        ("en", "ta"): {"repo": "Helsinki-NLP/opus-mt-en-dra", "prefix": ">>tam<<"},
        ("en", "te"): {"repo": "Helsinki-NLP/opus-mt-en-dra", "prefix": ">>tel<<"},
        ("en", "kn"): {"repo": "Helsinki-NLP/opus-mt-en-dra", "prefix": ">>kan<<"},
        ("en", "ml"): {"repo": "Helsinki-NLP/opus-mt-en-dra", "prefix": ">>mal<<"},
        ("ta", "en"): {"repo": "Helsinki-NLP/opus-mt-dra-en", "prefix": None},
        ("te", "en"): {"repo": "Helsinki-NLP/opus-mt-dra-en", "prefix": None},
        ("kn", "en"): {"repo": "Helsinki-NLP/opus-mt-dra-en", "prefix": None},
        ("ml", "en"): {"repo": "Helsinki-NLP/opus-mt-dra-en", "prefix": None},
        ("en", "hi"): {"repo": "Helsinki-NLP/opus-mt-en-inc", "prefix": ">>hin<<"},
        ("en", "mr"): {"repo": "Helsinki-NLP/opus-mt-en-inc", "prefix": ">>mar<<"},
        ("en", "gu"): {"repo": "Helsinki-NLP/opus-mt-en-inc", "prefix": ">>guj<<"},
        ("en", "bn"): {"repo": "Helsinki-NLP/opus-mt-en-inc", "prefix": ">>ben<<"},
        ("en", "ur"): {"repo": "Helsinki-NLP/opus-mt-en-inc", "prefix": ">>urd<<"},
        ("en", "pa"): {"repo": "Helsinki-NLP/opus-mt-en-inc", "prefix": ">>pan_Guru<<"},
        ("hi", "en"): {"repo": "Helsinki-NLP/opus-mt-inc-en", "prefix": None},
        ("mr", "en"): {"repo": "Helsinki-NLP/opus-mt-inc-en", "prefix": None},
        ("gu", "en"): {"repo": "Helsinki-NLP/opus-mt-inc-en", "prefix": None},
        ("bn", "en"): {"repo": "Helsinki-NLP/opus-mt-inc-en", "prefix": None},
        ("ur", "en"): {"repo": "Helsinki-NLP/opus-mt-inc-en", "prefix": None},
        ("pa", "en"): {"repo": "Helsinki-NLP/opus-mt-inc-en", "prefix": None},
        ("en", "es"): {"repo": "Helsinki-NLP/opus-mt-en-es", "prefix": None},
        ("es", "en"): {"repo": "Helsinki-NLP/opus-mt-es-en", "prefix": None},
        ("en", "fr"): {"repo": "Helsinki-NLP/opus-mt-en-fr", "prefix": None},
        ("fr", "en"): {"repo": "Helsinki-NLP/opus-mt-fr-en", "prefix": None},
        ("en", "de"): {"repo": "Helsinki-NLP/opus-mt-en-de", "prefix": None},
        ("de", "en"): {"repo": "Helsinki-NLP/opus-mt-de-en", "prefix": None},
        ("es", "fr"): {"repo": "Helsinki-NLP/opus-mt-es-fr", "prefix": None},
        ("fr", "es"): {"repo": "Helsinki-NLP/opus-mt-fr-es", "prefix": None},
        ("es", "de"): {"repo": "Helsinki-NLP/opus-mt-es-de", "prefix": None},
        ("de", "es"): {"repo": "Helsinki-NLP/opus-mt-de-es", "prefix": None},
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu" and torch.get_num_threads() > 2:
            torch.set_num_threads(2)

        self.cache_dir = "data/models/huggingface"
        os.makedirs(self.cache_dir, exist_ok=True)

        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Tuple[Any, Optional[str]]] = {}

    def _get_lang_code(self, lang_code: str) -> str:
        code = lang_code.lower().strip()
        return self.LANG_MAP.get(code, code)

    def supports_pair(self, src_lang: str, tgt_lang: str) -> bool:
        src_code = self._get_lang_code(src_lang)
        tgt_code = self._get_lang_code(tgt_lang)
        
        if src_code == tgt_code:
            return True
        if (src_code, tgt_code) in self.PAIR_CONFIG:
            return True
        return (src_code, "en") in self.PAIR_CONFIG and ("en", tgt_code) in self.PAIR_CONFIG

    def _get_pair_info(self, src_code: str, tgt_code: str) -> Tuple[str, Optional[str]]:
        pair = (src_code, tgt_code)
        if pair in self.PAIR_CONFIG:
            return self.PAIR_CONFIG[pair]["repo"], self.PAIR_CONFIG[pair]["prefix"]
        raise ValueError(f"Language pair ({src_code}, {tgt_code}) is not supported by MarianEngine.")

    def _load_model_and_tokenizer(self, repo_name: str, prefix: Optional[str]) -> Tuple[Any, Any, Optional[str]]:
        if repo_name not in self._models:
            # LRU Eviction: purge oldest model ONLY when cache limit is exceeded
            if len(self._models) >= self.MAX_CACHED_MODELS:
                oldest_repo = next(iter(self._models))
                del self._models[oldest_repo]
                del self._tokenizers[oldest_repo]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                tokenizer = AutoTokenizer.from_pretrained(repo_name, cache_dir=self.cache_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(repo_name, cache_dir=self.cache_dir)
                model.eval()

                if self.device != "cpu":
                    model = model.to(self.device)

                self._models[repo_name] = model
                self._tokenizers[repo_name] = (tokenizer, prefix)
            except Exception as e:
                raise RuntimeError(f"Failed to load Marian model '{repo_name}': {e}") from e

        tokenizer, prefix_tag = self._tokenizers[repo_name]
        return self._models[repo_name], tokenizer, prefix_tag

    def _execute_translation(self, texts: List[str], repo_name: str, prefix: Optional[str]) -> List[str]:
        model, tokenizer, prefix_tag = self._load_model_and_tokenizer(repo_name, prefix)

        prepared_texts = [f"{prefix_tag}{t}" for t in texts] if prefix_tag else texts

        encoded = tokenizer(
            prepared_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated_tokens = model.generate(
                **encoded,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=128,
                min_new_tokens=1,
                num_beams=1,
                use_cache=True
            )

        raw_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        cleaned_outputs = []
        for text in raw_outputs:
            cleaned = re.sub(r'<\s*dnt\s*_\s*(\d+)\s*>', r'<dnt_\1>', text, flags=re.IGNORECASE)
            cleaned_outputs.append(cleaned)

        return cleaned_outputs

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text.strip():
            return ""
        return self.batch_translate([text], src_lang, tgt_lang)[0]

    def batch_translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        if not texts:
            return []

        src_code = self._get_lang_code(src_lang)
        tgt_code = self._get_lang_code(tgt_lang)

        if src_code == tgt_code:
            return texts

        pair = (src_code, tgt_code)
        if pair not in self.PAIR_CONFIG:
            if src_code != "en" and tgt_code != "en":
                intermediate = self.batch_translate(texts, src_code, "en")
                return self.batch_translate(intermediate, "en", tgt_code)
            raise ValueError(f"Language pair ({src_code}, {tgt_code}) is not supported by MarianEngine.")

        repo_name, prefix = self._get_pair_info(src_code, tgt_code)
        return self._execute_translation(texts, repo_name, prefix)

    def get_model_family(self) -> str:
        return "Model_Family_2_Marian"