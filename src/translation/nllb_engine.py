import os
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.translation.base import BaseTranslationEngine


class NLLBEngine(BaseTranslationEngine):
    """
    Primary Multilingual Translation Engine using NLLB-200.
    Handles all required target languages and non-English directions.
    """

    LANG_MAP: Dict[str, str] = {
        # Indic
        "hi": "hin_Deva",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "bn": "ben_Beng",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "pa": "pan_Guru",
        "ur": "urd_Arab",
        # European & Global
        "en": "eng_Latn",
        "es": "spa_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "pt": "por_Latn",
        "id": "ind_Latn",
    }

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(4)

        cache_dir = "data/models/huggingface"
        os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)

    def _get_flores_code(self, lang_code: str) -> str:
        code = self.LANG_MAP.get(lang_code.lower().strip())
        if not code:
            return lang_code.strip()
        return code

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text.strip():
            return ""
        return self.batch_translate([text], src_lang, tgt_lang)[0]

    def batch_translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        if not texts:
            return []

        src_code = self._get_flores_code(src_lang)
        tgt_code = self._get_flores_code(tgt_lang)

        self.tokenizer.src_lang = src_code

        encoded = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if hasattr(self.tokenizer, "lang_code_to_id") and tgt_code in self.tokenizer.lang_code_to_id:
            tgt_lang_id = self.tokenizer.lang_code_to_id[tgt_code]
        else:
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            if tgt_lang_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Unsupported NLLB target language code: '{tgt_code}'")

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=tgt_lang_id,
                max_length=256,
                num_beams=1,
            )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def get_model_family(self) -> str:
        return "Model_Family_1_NLLB"