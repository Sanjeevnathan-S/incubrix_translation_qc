import re
from typing import Dict, List, Tuple, Any, Optional, Type


class TextMasker:
    """
    Production-grade DNT (Do Not Translate) processor.
    Implements non-overlapping span masking, sequential unique placeholders,
    subword-distortion recovery, strict validation, and deterministic restoration.
    """

    PATTERNS = [
        # 1. Template Placeholders
        ("TEMPLATE", r'\{\{[^{}]+\}\}|\{[^{}]+\}|%[sdifxX]|\$\{.*?\}'),
        
        # 2. URLs & URIs (excludes trailing sentence punctuation)
        ("URL", r'https?://[^\s<>"{}|\^~\[\]`]+[^\s<>"{}|\^~\[\]`.,!?:;]'),
        
        # 3. Email Addresses
        ("EMAIL", r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        
        # 4. UUIDs and Hashes
        ("UUID", r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'),
        ("HASH", r'\b[0-9a-fA-F]{32,64}\b'),
        
        # 5. Usernames / Handles (@user)
        ("MENTION", r'@[A-Za-z0-9_]+'),
        
        # 6. Hashtags (supports Unicode and Indic scripts)
        ("HASHTAG", r'#[A-Za-z0-9_\u00C0-\u024F\u0900-\u0D7F]+'),
        
        # 7. Currency, Timestamps, Numbers, Units
        ("CURRENCY_UNIT_NUM", 
         r'(?:[\$\€\£\₹\¥]\s*\d+(?:[\.,]\d+)*|\b\d+(?:[\.,:\/-]\d+)*(?:\s*(?:USD|EUR|INR|GBP|kg|g|km|m|cm|mm|am|pm|AM|PM|hrs|mins|secs|%))?)')
    ]

    PLACEHOLDER_REPAIR_REGEX = re.compile(r'_+\s*dnt\s*_+\s*(\d+)\s*_+', re.IGNORECASE)

    @classmethod
    def _normalize_placeholders(cls, text: str) -> str:
        """Repairs subword spacing and casing alterations introduced during model generation."""
        if not text:
            return ""

        def _repair_match(match: re.Match) -> str:
            idx = int(match.group(1))
            return f"__DNT_{idx:03d}__"

        return cls.PLACEHOLDER_REPAIR_REGEX.sub(_repair_match, text)

    @classmethod
    def mask(
        cls, 
        text: str, 
        dnt_terms: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, str]]:
        if not text or not text.strip():
            return text, {}

        spans: List[Tuple[int, int, str]] = []

        if dnt_terms:
            sorted_dnt = sorted(set(filter(None, dnt_terms)), key=len, reverse=True)
            for term in sorted_dnt:
                pattern = re.escape(term)
                for match in re.finditer(pattern, text):
                    start, end = match.span()
                    if not cls._has_overlap(start, end, spans):
                        spans.append((start, end, match.group(0)))

        for _, pattern in cls.PATTERNS:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                if not cls._has_overlap(start, end, spans):
                    spans.append((start, end, match.group(0)))

        spans.sort(key=lambda x: x[0])

        mapping: Dict[str, str] = {}
        masked_chunks = []
        last_idx = 0
        counter = 1

        for start, end, original_val in spans:
            masked_chunks.append(text[last_idx:start])
            placeholder = f"__DNT_{counter:03d}__"
            mapping[placeholder] = original_val
            masked_chunks.append(f" {placeholder} ")
            last_idx = end
            counter += 1

        masked_chunks.append(text[last_idx:])
        masked_text = "".join(masked_chunks)
        masked_text = re.sub(r'[ \t]+', ' ', masked_text).strip()
        return masked_text, mapping

    @staticmethod
    def _has_overlap(start: int, end: int, spans: List[Tuple[int, int, str]]) -> bool:
        for s_start, s_end, _ in spans:
            if max(start, s_start) < min(end, s_end):
                return True
        return False

    @classmethod
    def validate(
        cls, 
        translated_masked_text: str, 
        mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validates placeholders on the raw masked translation output from the model."""
        if not mapping:
            return {"status": "success", "valid": True, "missing": [], "unexpected": []}

        normalized_text = cls._normalize_placeholders(translated_masked_text)
        expected_placeholders = set(mapping.keys())
        found_placeholders = re.findall(r'__DNT_\d+__', normalized_text)
        
        counts: Dict[str, int] = {}
        for ph in found_placeholders:
            counts[ph] = counts.get(ph, 0) + 1

        missing = [ph for ph in expected_placeholders if counts.get(ph, 0) == 0]
        unexpected = [ph for ph, cnt in counts.items() if ph not in expected_placeholders or cnt > 1]

        passed = len(missing) == 0 and len(unexpected) == 0
        return {
            "status": "success" if passed else "review_required",
            "valid": passed,
            "missing": sorted(missing),
            "unexpected": sorted(unexpected)
        }

    @classmethod
    def unmask(
        cls, 
        translated_text: str, 
        mapping: Dict[str, str]
    ) -> str:
        if not translated_text or not mapping:
            return translated_text or ""

        normalized_text = cls._normalize_placeholders(translated_text)
        unmasked = normalized_text
        for placeholder, original_value in mapping.items():
            unmasked = unmasked.replace(placeholder, original_value)

        unmasked = re.sub(r'\s+([.,!?:;])', r'\1', unmasked)
        return re.sub(r'[ \t]+', ' ', unmasked).strip()

class MarianTextMasker(TextMasker):
    """
    Version 8: Angle Brackets <DNT001>, <DNT002>
    """
    PLACEHOLDER_REPAIR_REGEX = re.compile(r'<\s*DNT\s*(\d+)\s*>', re.IGNORECASE)
    PLACEHOLDER_FIND_REGEX = re.compile(r'<DNT\d{3}>')

    @classmethod
    def _format_placeholder(cls, counter: int) -> str:
        return f"<DNT{counter:03d}>"

    @classmethod
    def _normalize_placeholders(cls, text: str) -> str:
        if not text:
            return ""

        def _repair_match(match: re.Match) -> str:
            idx = int(match.group(1))
            return f" <DNT{idx:03d}> "

        repaired = cls.PLACEHOLDER_REPAIR_REGEX.sub(_repair_match, text)
        return re.sub(r'[ \t]+', ' ', repaired).strip()

def get_masker(engine_name: Optional[str] = None) -> Type[TextMasker]:
    """Dynamically resolves the engine-appropriate masker class."""
    if engine_name and "marian" in engine_name.lower():
        return MarianTextMasker
    return TextMasker