from diffproc import Passage, PipelineConfig, process_edit
from diffproc.tests.fixtures import make_contrib
from diffproc.usefulness import check_usefulness

LONG = (
    "The committee reviewed the proposal carefully before voting and then published a detailed explanation of every change for the public record today. "
    "Members asked several direct questions about the budget and waited for a complete written answer from the chair before they agreed to adjourn the meeting."
)


def _passage(prose: str, *, kind: str = "new", inserted: int | None = None, n_sentences: int = 2) -> Passage:
    return Passage(
        revid=1,
        parentid=0,
        pageid=1,
        ns=0,
        ns_group="content",
        timestamp="2026-01-05T12:34:00Z",
        user="Alice",
        raw_wikitext=prose,
        prose=prose,
        n_sentences=n_sentences,
        n_chars=len(prose),
        inserted_chars_total=len(prose) if inserted is None else inserted,
        kind=kind,
    )


def test_long_prose_is_useful():
    result = process_edit(make_contrib(), "Earlier text stays in place.\n", "Earlier text stays in place.\n\n" + LONG + "\n")
    assert result.accepted_stage1
    assert [passage.kind for passage in result.passages] == ["new"]
    assert result.useful is True


def test_infobox_dominates_prose_ratio():
    infobox = "{{Infobox person\n" + "\n".join(f"| field{i} = value{i}" for i in range(400)) + "\n}}"
    old = infobox + "\n\nEarlier text stays in place.\n"
    new = infobox.replace("value0", "valueX") + "\n\nEarlier text stays in place.\n\n" + LONG + "\n"
    result = process_edit(make_contrib(sizediff=5000), old, new)
    assert [passage.kind for passage in result.passages] == ["new"]
    assert result.useful is False
    assert result.usefulness_reason == "low_prose_ratio"


def test_modified_passages_do_not_count():
    useful, reason = check_usefulness([_passage(LONG, kind="modified")], "content")
    assert useful is False
    assert reason == "too_little_prose"


def test_too_few_sentences():
    prose = LONG.split(". ", 1)[0] + ". " + ("padding " * 40)
    useful, reason = check_usefulness([_passage(prose, n_sentences=1)], "content")
    assert useful is False
    assert reason == "too_few_sentences"


def test_all_caps_and_repeated_run():
    caps = ("THE COMMITTEE REVIEWED THE PROPOSAL CAREFULLY BEFORE VOTING TODAY. " * 8).strip()
    useful, reason = check_usefulness([_passage(caps)], "content")
    assert reason == "all_caps"
    assert useful is False

    noisy = LONG + " " + ("a" * 10)
    useful, reason = check_usefulness([_passage(noisy)], "content")
    assert reason == "repeated_run"
    assert useful is False


def test_low_letter_ratio():
    digits = ("1234567890 " * 40).strip() + ". Further digits follow this marker now."
    useful, reason = check_usefulness([_passage(digits)], "content")
    assert reason == "low_letter_ratio"
    assert useful is False
    assert PipelineConfig().min_letter_ratio == 0.7
