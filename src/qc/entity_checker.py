from typing import Dict, Any


class EntityChecker:
    """
    Deterministic Entity & Anomaly Checker.
    Validates that all masked URLs, hashtags, numbers, and Do-Not-Translate (DNT) terms
    survived translation without omission or corruption.
    """

    def check_preservation(self, original_mapping: Dict[str, str], translated_text: str) -> Dict[str, Any]:
        """Scans translated output for the presence of all original entity values (with case-insensitive fallback)."""
        if not original_mapping:
            return {
                "passed": True,
                "score": 1.0,
                "missing_entities": [],
                "total_entities": 0
            }

        translated_text_lower = translated_text.lower()
        missing = []

        for placeholder, original_value in original_mapping.items():
            val_str = str(original_value)
            # Check exact match first, fallback to case-insensitive match
            if val_str not in translated_text and val_str.lower() not in translated_text_lower:
                missing.append({
                    "placeholder": placeholder,
                    "expected": original_value
                })

        total = len(original_mapping)
        missing_count = len(missing)
        passed = (missing_count == 0)
        score = round((total - missing_count) / total, 2) if total > 0 else 1.0

        return {
            "passed": passed,
            "score": score,
            "missing_entities": missing,
            "total_entities": total
        }