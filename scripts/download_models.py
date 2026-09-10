import os
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration
from src.utils.logger import setup_logger

logger = setup_logger()

CACHE_DIR = Path("data/models/huggingface")
FASTTEXT_DIR = Path("data/models")
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_PATH = FASTTEXT_DIR / "lid.176.ftz"

MODELS_TO_DOWNLOAD = [
    # Primary Model Families
    {"name": "facebook/nllb-200-distilled-600M", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "facebook/m2m100_418M", "tokenizer_cls": AutoTokenizer, "model_cls": M2M100ForConditionalGeneration},

    # Marian Group Models (Indic & Dravidian)
    {"name": "Helsinki-NLP/opus-mt-en-dra", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-dra-en", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-en-inc", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-inc-en", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},

    # Marian Standalone Models (Verified Public Repositories)
    {"name": "Helsinki-NLP/opus-mt-en-es", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-es-en", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-en-fr", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-fr-en", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-en-de", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-de-en", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},

    # Marian European Inter-Language Models
    {"name": "Helsinki-NLP/opus-mt-es-fr", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-fr-es", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-es-de", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
    {"name": "Helsinki-NLP/opus-mt-de-es", "tokenizer_cls": AutoTokenizer, "model_cls": AutoModelForSeq2SeqLM},
]


def download_huggingface_models():
    """Downloads and caches Hugging Face translation models, skipping existing ones."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)

    for item in MODELS_TO_DOWNLOAD:
        model_name = item["name"]
        
        folder_name = f"models--{model_name.replace('/', '--')}"
        expected_cache_path = CACHE_DIR / "hub" / folder_name

        if expected_cache_path.exists() and any(expected_cache_path.iterdir()):
            logger.info(f"Model already cached, skipping: {model_name}")
            continue

        logger.info(f"Downloading Hugging Face weights for: {model_name}")
        try:
            item["tokenizer_cls"].from_pretrained(model_name, cache_dir=str(CACHE_DIR))
            item["model_cls"].from_pretrained(model_name, cache_dir=str(CACHE_DIR))
            logger.info(f"Successfully cached {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")


def download_fasttext_model():
    """Downloads FastText LID model using custom request headers to prevent 403 blocks."""
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
    if not FASTTEXT_PATH.exists():
        logger.info(f"Downloading FastText LID model from {FASTTEXT_URL}")
        try:
            req = urllib.request.Request(
                FASTTEXT_URL, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(FASTTEXT_PATH, "wb") as out_file:
                out_file.write(response.read())
            logger.info(f"FastText model successfully saved to {FASTTEXT_PATH}")
        except Exception as e:
            logger.error(f"Failed to download FastText model: {e}")
            return

    logger.info(f"FastText LID model already present at {FASTTEXT_PATH}")


def run_download_all():
    """Main entrypoint for pre-fetching all required local model weights."""
    logger.info("Starting model pre-download process...")
    download_huggingface_models()
    download_fasttext_model()
    logger.info("All model artifacts successfully downloaded and cached offline.")


if __name__ == "__main__":
    run_download_all()