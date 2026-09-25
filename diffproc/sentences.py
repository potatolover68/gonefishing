import hashlib
import re

_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT.split(text) if part.strip()]


def normalise(text: str) -> str:
    folded = text.casefold()
    folded = _PUNCT.sub(" ", folded)
    return _WS.sub(" ", folded).strip()


def tokens(text: str) -> list[str]:
    normalised = normalise(text)
    if not normalised:
        return []
    return normalised.split(" ")


def sentence_hash(text: str) -> bytes:
    return hashlib.blake2s(normalise(text).encode("utf-8"), digest_size=8).digest()


def usable_sentences(text: str, min_words: int) -> list[str]:
    return [
        sentence
        for sentence in split_sentences(text)
        if len(tokens(sentence)) >= min_words
    ]
