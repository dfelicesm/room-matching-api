import re
import unicodedata
from typing import List, Optional

import numpy as np

NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

STOPWORDS = {
    "and",
    "with",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "at",
    "to",
    "by",
    "on",
    "or",
}

REPLACEMENTS = [
    (r"&", " and "),
    (r"\+", " and "),
    (r"\bking bed(s)?\b", "king bed"),
    (r"\btwin bed(s)?\b", "twin bed"),
    (r"\bdouble bed(s)?\b", "double bed"),
    (r"\bqueen bed(s)?\b", "queen bed"),
]


def strip_accents(text: str) -> str:
    """Remove accents/diacritics from a string."""
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize(text: str) -> str:
    """
    Lowercase, strip accents, apply replacements, remove punctuation,
    collapse spaces.
    """
    s = text.strip().lower()
    s = strip_accents(s)

    # number words -> digits
    for word, digit in NUMBER_WORDS.items():
        s = re.sub(rf"\b{word}\b", digit, s)

    # pattern replacements
    for pat, rep in REPLACEMENTS:
        s = re.sub(pat, rep, s)

    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_tokens(text: str) -> List[str]:
    """Normalize text and split into tokens (keeps stopwords)."""
    return normalize(text).split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from a list of tokens."""
    return [t for t in tokens if t not in STOPWORDS]


def tokenize(text: str) -> List[str]:
    """Normalize, split, and remove stopwords in one step."""
    return remove_stopwords(split_tokens(text))


def extract_room_type(text: str) -> Optional[str]:
    """Extract room type keyword (suite, apartment, loft, room)."""
    match = re.search(r"\b(house|suite|apartment|loft|room)\b", text)
    return match.group(1) if match else np.nan


def extract_bed_type(tokens: List[str]) -> Optional[str]:
    """Extract bed type (king, twin, double, queen) if followed by 'bed'."""
    bed_types = {"king", "twin", "double", "queen"}
    for i, tok in enumerate(tokens):
        if tok in bed_types and i + 1 < len(tokens) and tokens[i + 1] == "bed":
            return tok
    return np.nan


def extract_beds(tokens: List[str]) -> Optional[int]:
    """Extract number of beds (e.g., '2 twin bed' -> 2)."""
    bed_types = {"king", "twin", "double", "queen"}
    for i, tok in enumerate(tokens):
        if tok in bed_types and i > 0 and tokens[i - 1].isdigit():
            return int(tokens[i - 1])
    return np.nan


def extract_bedroom_count(tokens: List[str]) -> Optional[int]:
    """Extract number of bedrooms (e.g., '2 bedroom' -> 2)."""
    for i, tok in enumerate(tokens):
        if tok == "bedroom" and i > 0 and tokens[i - 1].isdigit():
            return int(tokens[i - 1])
    return np.nan
