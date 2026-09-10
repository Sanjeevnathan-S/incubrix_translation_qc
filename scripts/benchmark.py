import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import List, Dict, Any

from src.translation.nllb_engine import NLLBEngine
from src.translation.marian_engine import MarianEngine
from src.utils.masker import get_masker
from src.utils.metrics import calculate_bleu, calculate_chrf
from src.utils.logger import get_peak_memory_mb, setup_logger

logger = setup_logger()

DEFAULT_BENCHMARK_DATASET = [
    {
        "id": "bench_01",
        "src_lang": "en",
        "tgt_lang": "hi",
        "domain": "Customer Support & DNT",
        "dnt_terms": ["IncuBrix", "IncuBrix Express"],
        "source": "Your order #89201 has been dispatched via IncuBrix Express. Track it at https://incubrix.com/track.",
        "ref": "आपका ऑर्डर #89201 IncuBrix Express के माध्यम से भेज दिया गया है। इसे https://incubrix.com/track पर ट्रैक करें।"
    },
    {
        "id": "bench_02",
        "src_lang": "en",
        "tgt_lang": "ta",
        "domain": "Security & Timestamps",
        "dnt_terms": ["IST", "SecurityAlert"],
        "source": "Please update your password before 10:00 AM IST to prevent account suspension. #SecurityAlert",
        "ref": "கணக்கு இடைநிறுத்தப்படுவதைத் தடுக்க காலை 10:00 AM IST மணிக்குள் உங்கள் கடவுச்சொல்லைப் புதுப்பிக்கவும். #SecurityAlert"
    },
    {
        "id": "bench_03",
        "src_lang": "en",
        "tgt_lang": "es",
        "domain": "E-Commerce & Numbers",
        "dnt_terms": [],
        "source": "Get up to 50% discount on all wireless headsets today. Terms and conditions apply.",
        "ref": "Obtenga hasta un 50% de descuento en todos los auriculares inalámbricos hoy. Se aplican términos y condiciones."
    },
    {
        "id": "bench_04",
        "src_lang": "en",
        "tgt_lang": "fr",
        "domain": "Enterprise Policy",
        "dnt_terms": ["VPN"],
        "source": "All remote employees must connect through the corporate VPN when accessing internal servers.",
        "ref": "Tous les employés à distance doivent se connecter via le VPN d'entreprise pour accéder aux serveurs internes."
    },
    {
        "id": "bench_05",
        "src_lang": "hi",
        "tgt_lang": "ta",
        "domain": "Indic to Indic Non-English",
        "dnt_terms": [],
        "source": "कृपया अपनी यात्रा की पुष्टि करने के लिए 1 पर क्लिक करें।",
        "ref": "உங்கள் பயணத்தை உறுதிப்படுத்த 1 ஐக் கிளிக் செய்யவும்."
    },
    {
        "id": "bench_06",
        "src_lang": "fr",
        "tgt_lang": "es",
        "domain": "European Cross-Directional",
        "dnt_terms": [],
        "source": "Veuillez vérifier les détails de votre compte avant de procéder au paiement.",
        "ref": "Por favor verifique los detalles de su cuenta antes de proceder al pago."
    }
]


def load_benchmark_data(input_file: str = None) -> List[Dict[str, Any]]:
    if input_file and os.path.exists(input_file):
        logger.info(f"Loading custom benchmark dataset from: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            if input_file.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)
    logger.info("No custom file provided or file not found. Using embedded multi-domain test set.")
    return DEFAULT_BENCHMARK_DATASET


def evaluate_engine_performance(engine, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    bleu_scores = []
    chrf_scores = []
    processed_records = []
    dnt_passed_count = 0
    
    model_family = getattr(engine, "get_model_family", lambda: engine.__class__.__name__)()

    # Dynamically select engine-appropriate masker class
    masker = get_masker(model_family)

    # Warm-up phase: Run dummy inference across all distinct language directions
    # to eliminate cold-start loading, disk IO, and PyTorch runtime allocation.
    logger.info(f"Warming up engine model checkpoints for {model_family}...")
    unique_pairs = {(s["src_lang"], s["tgt_lang"]) for s in samples}
    for src, tgt in unique_pairs:
        if hasattr(engine, "supports_pair") and not engine.supports_pair(src, tgt):
            continue
        try:
            engine.translate("Warmup initialization string.", src_lang=src, tgt_lang=tgt)
        except Exception as e:
            logger.warning(f"Warmup pass skipped for pair ({src}->{tgt}): {e}")
    logger.info(f"Warm-up complete for {model_family}. Starting benchmark timing loop...")

    start_time = time.time()
    total_tokens = 0

    for sample in samples:
        source_text = sample["source"]
        ref_text = sample.get("ref", "")
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]
        dnt_terms = sample.get("dnt_terms", [])

        # Skip unsupported pair evaluation for Marian
        if hasattr(engine, "supports_pair") and not engine.supports_pair(src_lang, tgt_lang):
            logger.warning(f"Skipping sample {sample.get('id')} ({src_lang}->{tgt_lang}): Unsupported by {model_family}")
            continue

        # 1. Mask non-translatables using model-specific tags
        masked_text, mapping = masker.mask(source_text, dnt_terms=dnt_terms)

        # 2. Run engine inference with fallback safety per sample
        try:
            raw_output = engine.translate(masked_text, src_lang=src_lang, tgt_lang=tgt_lang)
        except Exception as e:
            logger.error(f"Inference failed for sample '{sample.get('id')}' on engine {model_family}: {e}")
            raw_output = ""

        # 3. Validate placeholder integrity using engine-specific regex patterns
        val_res = masker.validate(raw_output, mapping)
        if val_res.get("status") == "success":
            dnt_passed_count += 1

        # 4. Unmask back to original entities using engine-appropriate normalization
        final_output = masker.unmask(raw_output, mapping)

        total_tokens += len(final_output.split())

        b_score = calculate_bleu(final_output, ref_text) if ref_text and final_output else 0.0
        c_score = calculate_chrf(final_output, ref_text) if ref_text and final_output else 0.0

        bleu_scores.append(b_score)
        chrf_scores.append(c_score)

        processed_records.append({
            "id": sample.get("id"),
            "domain": sample.get("domain", "general"),
            "masker_used": masker.__name__,
            "source": source_text,
            "masked_input": masked_text,
            "raw_output": raw_output,
            "final_translation": final_output,
            "validation": val_res,
            "reference": ref_text,
            "bleu": b_score,
            "chrf": c_score
        })

    elapsed_sec = time.time() - start_time
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
    latency_ms = (elapsed_sec / len(samples)) * 1000 if samples else 0.0
    throughput = total_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
    dnt_pass_rate = (dnt_passed_count / len(samples)) * 100 if samples else 0.0

    return {
        "model_family": model_family,
        "masker_used": masker.__name__,
        "total_samples": len(samples),
        "avg_bleu": round(avg_bleu, 2),
        "avg_chrf": round(avg_chrf, 2),
        "dnt_pass_rate": f"{dnt_pass_rate:.1f}%",
        "latency_per_segment_ms": round(latency_ms, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "peak_ram_mb": get_peak_memory_mb(),
        "records": processed_records
    }


def run_benchmark(input_path: str = None, output_path: str = "data/outputs/benchmark_results.json"):
    dataset = load_benchmark_data(input_path)
    logger.info(f"Loaded {len(dataset)} evaluation samples. Starting comparative execution...")

    # 1. Evaluate Model Family 1 (NLLB-200)
    logger.info("Testing Model Family 1: NLLB-200...")
    f1_results = {"total_samples": len(dataset), "avg_bleu": 0.0, "avg_chrf": 0.0, "dnt_pass_rate": "0.0%", "latency_per_segment_ms": 0.0, "throughput_tokens_per_sec": 0.0, "peak_ram_mb": 0.0}
    try:
        nllb = NLLBEngine()
        f1_results = evaluate_engine_performance(nllb, dataset)
        del nllb
        gc.collect()
    except Exception as e:
        logger.error(f"Failed to initialize or evaluate NLLB Engine: {e}")

    # 2. Evaluate Model Family 2 (M2M-100 / Marian)
    logger.info("Testing Model Family 2: M2M-100 / Marian...")
    f2_results = {"total_samples": len(dataset), "avg_bleu": 0.0, "avg_chrf": 0.0, "dnt_pass_rate": "0.0%", "latency_per_segment_ms": 0.0, "throughput_tokens_per_sec": 0.0, "peak_ram_mb": 0.0}
    try:
        marian = MarianEngine()
        f2_results = evaluate_engine_performance(marian, dataset)
        del marian
        gc.collect()
    except Exception as e:
        logger.error(f"Failed to initialize or evaluate Marian Engine: {e}")

    # 3. Print Deliverable Comparison Table
    print("\n" + "=" * 80)
    print("                    CPU MODEL COMPARISON BENCHMARK TABLE                    ")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Model Family 1 (NLLB)':<22} | {'Model Family 2 (Marian)':<22}")
    print("-" * 80)
    print(f"{'Total Samples Evaluated':<30} | {f1_results['total_samples']:<22} | {f2_results['total_samples']:<22}")
    print(f"{'Avg BLEU Score':<30} | {f1_results['avg_bleu']:<22} | {f2_results['avg_bleu']:<22}")
    print(f"{'Avg chrF Score':<30} | {f1_results['avg_chrf']:<22} | {f2_results['avg_chrf']:<22}")
    print(f"{'DNT Preservation Pass Rate':<30} | {f1_results['dnt_pass_rate']:<22} | {f2_results['dnt_pass_rate']:<22}")
    print(f"{'Latency (ms/segment)':<30} | {f1_results['latency_per_segment_ms']:<22} | {f2_results['latency_per_segment_ms']:<22}")
    print(f"{'Throughput (tokens/sec)':<30} | {f1_results['throughput_tokens_per_sec']:<22} | {f2_results['throughput_tokens_per_sec']:<22}")
    print(f"{'Peak Memory Usage (MB)':<30} | {f1_results['peak_ram_mb']:<22} | {f2_results['peak_ram_mb']:<22}")
    print("=" * 80 + "\n")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"family_1": f1_results, "family_2": f2_results}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved full benchmark report to: {out_file}")


if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(input_path=file_arg)