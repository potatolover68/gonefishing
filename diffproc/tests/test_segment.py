from diffproc.segment import diff_paragraphs, split_paragraphs


def test_infobox_blank_line_stays_one_paragraph():
    text = "{{Infobox person\n| name = Ada\n\n| birth = 1815\n}}\n\nShe wrote notes.\n"
    paragraphs = split_paragraphs(text)
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("{{Infobox")
    assert "birth" in paragraphs[0]
    assert paragraphs[1] == "She wrote notes."


def test_whitespace_only_change_is_unchanged():
    added, removed = diff_paragraphs(
        "The committee reviewed the proposal carefully before voting.\n",
        "The committee reviewed the proposal\ncarefully before voting.\n",
    )
    assert added == []
    assert removed == []


def test_duplicate_paragraph_multiset():
    added, removed = diff_paragraphs(
        "Alpha sentence one is here.\n\nAlpha sentence one is here.\n",
        "Alpha sentence one is here.\n",
    )
    assert added == []
    assert len(removed) == 1
