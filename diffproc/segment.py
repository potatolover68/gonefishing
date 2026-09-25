import hashlib
import re
from collections import Counter

_WS = re.compile(r"\s+")


def paragraph_hash(text: str) -> bytes:
    collapsed = _WS.sub(" ", text).strip()
    return hashlib.blake2s(collapsed.encode("utf-8"), digest_size=8).digest()


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    start = 0
    i = 0
    n = len(text)
    template_depth = 0
    table_depth = 0
    while i < n:
        if text.startswith("{{", i):
            template_depth += 1
            i += 2
            continue
        if text.startswith("}}", i):
            template_depth = max(0, template_depth - 1)
            i += 2
            continue
        if text.startswith("{|", i):
            table_depth += 1
            i += 2
            continue
        if text.startswith("|}", i):
            table_depth = max(0, table_depth - 1)
            i += 2
            continue
        if template_depth == 0 and table_depth == 0 and text[i] == "\n":
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "\n":
                para = text[start:i].strip("\n")
                if para.strip():
                    parts.append(para)
                start = j + 1
                i = start
                continue
        i += 1
    tail = text[start:].strip("\n")
    if tail.strip():
        parts.append(tail)
    return parts


def _extras(paragraphs: list[str], other_counts: Counter) -> list[str]:
    seen: Counter = Counter()
    extras: list[str] = []
    for paragraph in paragraphs:
        digest = paragraph_hash(paragraph)
        seen[digest] += 1
        if seen[digest] > other_counts[digest]:
            extras.append(paragraph)
    return extras


def diff_paragraphs(old: str, new: str) -> tuple[list[str], list[str]]:
    old_paragraphs = split_paragraphs(old)
    new_paragraphs = split_paragraphs(new)
    old_counts = Counter(paragraph_hash(paragraph) for paragraph in old_paragraphs)
    new_counts = Counter(paragraph_hash(paragraph) for paragraph in new_paragraphs)
    added = _extras(new_paragraphs, old_counts)
    removed = _extras(old_paragraphs, new_counts)
    return added, removed
