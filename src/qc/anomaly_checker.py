from typing import Dict, Any


class AnomalyChecker:
    """Detects translation degeneration, repetitive n-gram loops, source copying, and length anomalies."""

    def check_anomalies(self, text: str, original_src: str = "") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"passed": False, "reason": "empty_output"}

        words = text.split()
        src_words = original_src.split() if original_src else []

        # 1. Raw Source Copy Check (for segments > 3 words)
        if original_src and text.strip().lower() == original_src.strip().lower() and len(src_words) > 3:
            return {"passed": False, "reason": "untranslated_source_copy"}

        # 2. Extreme Length Ratio Check
        if src_words:
            ratio = len(words) / max(len(src_words), 1)
            if ratio < 0.20 or ratio > 4.0:
                return {"passed": False, "reason": f"extreme_length_ratio_{ratio:.2f}"}

        # 3. Consecutive word repetition check (4+ identical consecutive words)
        if len(words) >= 4:
            for i in range(len(words) - 3):
                if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                    return {"passed": False, "reason": "consecutive_word_loop"}

        # 4. Phrase repetition check (2-3 word n-grams repeating 3+ times consecutively)
        for ngram_size in (2, 3):
            if len(words) >= ngram_size * 3:
                ngrams = [" ".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
                for i in range(len(ngrams) - (ngram_size * 2)):
                    if ngrams[i] == ngrams[i + ngram_size] == ngrams[i + ngram_size * 2]:
                        return {"passed": False, "reason": "phrase_repetition_loop"}

        return {"passed": True, "reason": "none"}