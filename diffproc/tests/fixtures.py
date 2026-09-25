"""Hand-built diff cases. These lock expected kinds and cleaned prose."""

from diffproc.types import Contrib

TALK = (
    "I think the article should mention the earlier committee decision in full "
    "and explain why the editors rejected the shorter wording during the last review."
)
STANDARD_SIG = (
    " [[User:Example|Example]] ([[User talk:Example|talk]]) 12:34, 5 January 2026 (UTC)"
)
CUSTOM_SIG = (
    ' <span style="font-family:comic sans; color:red">[[User:Example|✦ Example ✦]]</span>'
    " 12:34, 5 January 2026 (UTC)"
)


def make_contrib(**overrides) -> Contrib:
    data = dict(
        revid=10,
        parentid=9,
        pageid=100,
        ns=0,
        title="Example",
        timestamp="2026-01-05T12:34:00Z",
        comment="",
        sizediff=800,
        tags=frozenset(),
        is_new=False,
        user="Alice",
    )
    data.update(overrides)
    return Contrib(**data)


# (name, contrib, old, new, reject_reason, kinds, require, forbid)
# History check is off: text missing from the parent is new, even if an
# earlier revision once contained it.
FIXTURES = [
    (
        "section_moved",
        make_contrib(),
        "== Alpha ==\n\nThe committee reviewed the proposal carefully before voting.\n\n== Beta ==\n\nAnother fully formed sentence lives in this section today.\n",
        "== Beta ==\n\nAnother fully formed sentence lives in this section today.\n\n== Alpha ==\n\nThe committee reviewed the proposal carefully before voting.\n",
        None,
        (),
        (),
        (),
    ),
    (
        "paragraph_moved_one_word",
        make_contrib(),
        "The committee reviewed the proposal carefully before voting.\n\nAnother fully formed sentence lives in this section today.\n",
        "Another fully formed sentence lives in this section today.\n\nThe committee reviewed the proposal carefully before the vote.\n",
        None,
        ("modified",),
        ("before the vote",),
        (),
    ),
    (
        "citation_reformat",
        make_contrib(),
        "The committee reviewed the proposal carefully before voting.<ref>Smith 2020</ref>\n",
        'The committee reviewed the proposal carefully before voting.<ref name="s">Smith 2020</ref>\n',
        None,
        (),
        (),
        ("Smith", "<ref"),
    ),
    (
        "copyedit_three_sentences",
        make_contrib(),
        "The committee reviewed the proposal carefully before voting. Members asked several direct questions about the budget. The chair recorded each answer in the minutes.",
        "The committee reviewed the proposal carefully before the vote. Members asked several direct questions about funding. The chair recorded every answer in the minutes.",
        None,
        ("modified",),
        ("before the vote", "about funding", "every answer"),
        (),
    ),
    (
        "sentence_split",
        make_contrib(),
        "The committee reviewed the proposal carefully before the members cast the final vote.",
        "The committee reviewed the proposal carefully before the members. The members cast the final vote today.",
        None,
        ("modified",),
        ("before the members", "final vote today"),
        (),
    ),
    (
        "new_paragraph_refs_links",
        make_contrib(),
        "The harbour has served local fishing boats for many decades.\n",
        "The harbour has served local fishing boats for many decades.\n\n"
        "The council published a report on [[Coastal erosion|the erosion of the coast]] after the winter storms."
        "<ref>Harbour Board 2024</ref> Residents asked for a clearer plan.\n",
        None,
        ("new",),
        ("the erosion of the coast", "clearer plan"),
        ("<ref", "Harbour Board", "Coastal erosion", "[["),
    ),
    (
        "talk_standard_signature",
        make_contrib(ns=1, title="Talk:Example", sizediff=200),
        "The article should mention the earlier decision.\n",
        "The article should mention the earlier decision.\n\n"
        + ":"
        + TALK
        + STANDARD_SIG
        + "\n",
        None,
        ("new",),
        (TALK,),
        ("Example", "12:34", "UTC", "User:"),
    ),
    (
        "talk_custom_signature",
        make_contrib(ns=1, title="Talk:Example", sizediff=200),
        "The article should mention the earlier decision.\n",
        "The article should mention the earlier decision.\n\n"
        + ":"
        + TALK
        + CUSTOM_SIG
        + "\n",
        None,
        ("new",),
        (TALK,),
        ("Example", "12:34", "UTC", "User:", "comic", "color", "span"),
    ),
    (
        "page_creation",
        make_contrib(parentid=0, is_new=True, sizediff=1200),
        None,
        "The committee reviewed the proposal carefully before voting. Members asked several direct questions about the budget. The chair recorded each answer in the minutes.",
        None,
        ("new",),
        ("before voting", "about the budget", "in the minutes"),
        (),
    ),
    (
        "page_creation_split_from",
        make_contrib(
            parentid=0, is_new=True, sizediff=1200, comment="split from [[Old page]]"
        ),
        None,
        "The committee reviewed the proposal carefully before voting. Members asked several direct questions about the budget.",
        "copied_or_restored",
        (),
        (),
        (),
    ),
    (
        "readded_text_absent_from_parent",
        make_contrib(),
        "The harbour has served local fishing boats for many decades.\n",
        "An earlier editor described the stone pier in careful detail before it was taken out.\n",
        None,
        ("new",),
        ("stone pier",),
        (),
    ),
    (
        "infobox_beside_prose",
        make_contrib(),
        "{{Infobox person\n| name = Ada\n\n| birth_date = 1815\n}}\n\n"
        "She wrote the first notes on the analytical engine and they were widely read.\n",
        "{{Infobox person\n| name = Ada\n\n| birth_date = 1815\n| death_date = 1852\n}}\n\n"
        "She wrote the first notes on the analytical engine and they were widely read.\n\n"
        "Later editors added a longer account of her work on the early computer.\n",
        None,
        ("new",),
        ("early computer",),
        ("Infobox", "birth_date", "Ada", "{{"),
    ),
]
