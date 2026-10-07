import re
from typing import Dict

HINGLISH_MAP: Dict[str, str] = {
    "khana": "food",
    "khaana": "food",
    "khaaana": "food",
    "accha": "good",
    "acha": "good",
    "achha": "good",
    "acchi": "good",
    "achi": "good",
    "bura": "bad",
    "buri": "bad",
    "bekar": "bad",
    "bakwas": "bad",
    "nahi": "not",
    "nahin": "not",
    "na": "not",
    "thanda": "cold",
    "thandi": "cold",
    "thand": "cold",
    "safai": "cleanliness",
    "saaf": "clean",
    "ganda": "dirty",
    "gandi": "dirty",
    "mehnga": "expensive",
    "sasta": "cheap",
    "badhiya": "great",
    "mast": "great",
    "shandar": "amazing",
    "lajawab": "delicious",
    "lajawaab": "delicious",
    "paisa": "money",
    "vasool": "worth",
    "barbad": "wasted",
}

NEGATION_MAP: Dict[str, str] = {
    r"\bnot\s+good\b": "bad",
    r"\bnot\s+clean\b": "dirty",
    r"\bnot\s+tasty\b": "distasteful",
    r"\bnot\s+worth\b": "worthless",
    r"\bnot\s+polite\b": "rude",
    r"\bnot\s+helpful\b": "unhelpful",
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed Alphanumerics
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Normalize raw review text across English, Hinglish, and Devanagari.

    Converts Hinglish and resolves negations to simplify downstream classification.

    Example:
      >>> normalize_text("Khana badhiya tha 😊 but room was not clean!")
      'food great tha but room was dirty!'

    Args:
        text: Raw review input string.

    Returns:
        str: Normalized, cleaned string.
    """
    if not text:
        return ""

    normalized = text.lower()
    normalized = EMOJI_PATTERN.sub("", normalized)

    def replace_hinglish(match: re.Match) -> str:
        word = match.group(0)
        return HINGLISH_MAP.get(word, word)

    word_pattern = re.compile(r"\b[a-z0-9\u0900-\u097f]+\b", re.IGNORECASE)
    normalized = word_pattern.sub(replace_hinglish, normalized)

    for pattern, replacement in NEGATION_MAP.items():
        normalized = re.sub(pattern, replacement, normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized
