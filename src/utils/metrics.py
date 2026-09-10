import sacrebleu

def calculate_bleu(hypothesis: str, reference: str) -> float:
    """Computes BLEU metric against reference text safely."""
    if not hypothesis or not reference or not hypothesis.strip() or not reference.strip():
        return 0.0
    try:
        bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
        return round(bleu.score, 2)
    except Exception:
        return 0.0

def calculate_chrf(hypothesis: str, reference: str) -> float:
    """Computes chrF metric against reference text safely."""
    if not hypothesis or not reference or not hypothesis.strip() or not reference.strip():
        return 0.0
    try:
        chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]])
        return round(chrf.score, 2)
    except Exception:
        return 0.0