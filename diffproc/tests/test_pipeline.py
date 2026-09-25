import pytest

from diffproc import process_edit
from diffproc.tests.fixtures import FIXTURES


@pytest.mark.parametrize(
    ("name", "contrib", "old", "new", "reject_reason", "kinds", "require", "forbid"),
    FIXTURES,
    ids=[case[0] for case in FIXTURES],
)
def test_fixture(name, contrib, old, new, reject_reason, kinds, require, forbid):
    result = process_edit(contrib, old, new)
    assert result.reject_reason == reject_reason
    assert tuple(passage.kind for passage in result.passages) == kinds
    prose = "\n".join(passage.prose for passage in result.passages)
    for snippet in require:
        assert snippet in prose
    for snippet in forbid:
        assert snippet not in prose
    if kinds == ("modified",) and name == "copyedit_three_sentences":
        assert result.passages[0].n_sentences == 3
    if kinds == ("modified",) and name == "sentence_split":
        assert result.passages[0].n_sentences == 2
    if name == "infobox_beside_prose":
        assert len(result.passages) == 1
        assert "Infobox" not in result.passages[0].raw_wikitext
    if reject_reason is None and kinds == ("new",) and contrib.ns == 1:
        assert result.useful is True
        assert result.usefulness_reason is None
