from collections import Counter

from diffproc.clean import clean_paragraph
from diffproc.config import PipelineConfig
from diffproc.sentences import sentence_hash, tokens, usable_sentences
from diffproc.types import Contrib, Passage


def _containment(candidate: str, removed: str) -> float:
    candidate_tokens = set(tokens(candidate))
    if not candidate_tokens:
        return 0.0
    removed_tokens = set(tokens(removed))
    return len(candidate_tokens & removed_tokens) / len(candidate_tokens)


def _best_containment(candidate: str, pool: list[str]) -> float:
    if not pool:
        return 0.0
    return max(_containment(candidate, removed) for removed in pool)


def _not_in_other(sentences: list[str], other_counts: Counter) -> list[str]:
    seen: Counter = Counter()
    kept: list[str] = []
    for sentence in sentences:
        digest = sentence_hash(sentence)
        seen[digest] += 1
        if seen[digest] > other_counts[digest]:
            kept.append(sentence)
    return kept


def extract_passages(
    contrib: Contrib,
    ns_group: str,
    added: list[str],
    removed: list[str],
    config: PipelineConfig,
) -> list[Passage]:
    inserted_chars_total = sum(len(paragraph) for paragraph in added)
    added_cleaned = [(raw, clean_paragraph(raw, ns_group, config)) for raw in added]
    removed_sentences: list[str] = []
    for raw in removed:
        cleaned = clean_paragraph(raw, ns_group, config)
        if cleaned:
            removed_sentences.extend(usable_sentences(cleaned, config.min_words))

    added_sentences: list[list[str]] = []
    added_counts: Counter = Counter()
    for _raw, cleaned in added_cleaned:
        sentences = usable_sentences(cleaned, config.min_words) if cleaned else []
        added_sentences.append(sentences)
        for sentence in sentences:
            added_counts[sentence_hash(sentence)] += 1

    removed_counts: Counter = Counter(
        sentence_hash(sentence) for sentence in removed_sentences
    )
    removed_pool = _not_in_other(removed_sentences, added_counts)

    passages: list[Passage] = []
    used_removed = Counter()
    for (raw, _cleaned), sentences in zip(added_cleaned, added_sentences):
        candidates: list[str] = []
        for sentence in sentences:
            digest = sentence_hash(sentence)
            if used_removed[digest] < removed_counts[digest]:
                used_removed[digest] += 1
                continue
            candidates.append(sentence)
        grouped = {"new": [], "modified": []}
        for sentence in candidates:
            if (
                _best_containment(sentence, removed_pool)
                >= config.containment_threshold
            ):
                grouped["modified"].append(sentence)
            else:
                grouped["new"].append(sentence)
        for kind in ("new", "modified"):
            kept = grouped[kind]
            if not kept:
                continue
            prose = " ".join(kept)
            passages.append(
                Passage(
                    revid=contrib.revid,
                    parentid=contrib.parentid,
                    pageid=contrib.pageid,
                    ns=contrib.ns,
                    ns_group=ns_group,
                    timestamp=contrib.timestamp,
                    user=contrib.user,
                    raw_wikitext=raw,
                    prose=prose,
                    n_sentences=len(kept),
                    n_chars=len(prose),
                    inserted_chars_total=inserted_chars_total,
                    kind=kind,
                )
            )
    return passages
