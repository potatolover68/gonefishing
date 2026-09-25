from dataclasses import dataclass


@dataclass(frozen=True)
class Contrib:
    revid: int
    parentid: int
    pageid: int
    ns: int
    title: str
    timestamp: str
    comment: str
    sizediff: int
    tags: frozenset[str]
    is_new: bool
    user: str


@dataclass(frozen=True)
class FilterDecision:
    accept: bool
    reason: str | None
    ns_group: str | None


@dataclass(frozen=True)
class Passage:
    revid: int
    parentid: int
    pageid: int
    ns: int
    ns_group: str
    timestamp: str
    user: str
    raw_wikitext: str
    prose: str
    n_sentences: int
    n_chars: int
    inserted_chars_total: int
    kind: str


@dataclass(frozen=True)
class ProcessResult:
    accepted_stage1: bool
    reject_reason: str | None
    passages: tuple[Passage, ...]
    useful: bool
    usefulness_reason: str | None
