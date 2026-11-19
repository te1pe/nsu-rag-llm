from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

WORD = re.compile(r"\w+")
CYRILLIC = re.compile(r"[а-я]")
LATIN = re.compile(r"[a-z]")

_russian = snowballstemmer.stemmer("russian")
_english = snowballstemmer.stemmer("english")


@lru_cache(maxsize=500_000)
def stem(word: str) -> str:
    has_cyrillic = bool(CYRILLIC.search(word))
    has_latin = bool(LATIN.search(word))
    if has_cyrillic and not has_latin:
        return _russian.stemWord(word)
    if has_latin and not has_cyrillic:
        return _english.stemWord(word)
    return word


def tokenize(text: str) -> list[str]:
    lowered = text.lower().replace("ё", "е")
    return [stem(word) for word in WORD.findall(lowered) if len(word) > 1]
