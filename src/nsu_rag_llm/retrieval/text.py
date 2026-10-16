"""Tokens for lexical search: lower-cased words, each stemmed in its own language.

CLAUDE.md asks for Snowball stemming "by the language of the fragment". The
fragments here are rarely one language: a Russian paragraph about `fork()`,
a Russian sentence quoting an English term. So the language is decided per
word, by its alphabet - Cyrillic words go to the Russian stemmer, Latin ones to
the English stemmer - and a query is tokenised the same way, so a word always
meets its own stem (ADR-022).
"""

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
    """One lower-cased word to its stem. Cached: a corpus repeats its words.

    A word mixing both alphabets, or none (a number, an identifier with
    digits), is kept as it is.
    """
    has_cyrillic = bool(CYRILLIC.search(word))
    has_latin = bool(LATIN.search(word))
    if has_cyrillic and not has_latin:
        return _russian.stemWord(word)
    if has_latin and not has_cyrillic:
        return _english.stemWord(word)
    return word


def tokenize(text: str) -> list[str]:
    """Stemmed tokens of a text, in order.

    `ё` is folded to `е` first: the corpus spells the same word both ways.
    Single characters are dropped - in this corpus they are almost always a
    variable in a formula or a one-letter preposition, and match everywhere.
    """
    lowered = text.lower().replace("ё", "е")
    return [stem(word) for word in WORD.findall(lowered) if len(word) > 1]
