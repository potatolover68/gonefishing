from diffproc import filter_contrib, groups_from_siteinfo, ns_group, process_edit
from diffproc.config import PipelineConfig
from diffproc.tests.fixtures import make_contrib


def test_namespace_groups():
    config = PipelineConfig()
    assert ns_group(0, config.content_ids) == "content"
    assert ns_group(4, config.content_ids) == "content"
    assert ns_group(118, config.content_ids) == "content"
    assert ns_group(1, config.content_ids) == "discussion"
    assert ns_group(119, config.content_ids) == "discussion"
    assert ns_group(2, config.content_ids) is None
    assert ns_group(100, config.content_ids) is None


def test_groups_from_siteinfo():
    namespaces = {
        "0": {"*": ""},
        "4": {"canonical": "Project", "*": "Wikipedia"},
        "118": {"canonical": "Draft", "*": "Draft"},
        "1": {"canonical": "Talk", "*": "Talk"},
        "2": {"canonical": "User", "*": "User"},
    }
    assert groups_from_siteinfo(namespaces) == frozenset({0, 4, 118})


def test_too_small():
    decision = filter_contrib(make_contrib(sizediff=100))
    assert decision.reason == "too_small"
    assert decision.ns_group == "content"


def test_rollback_and_tool_tags():
    assert (
        filter_contrib(make_contrib(tags=frozenset({"mw-rollback"}))).reason
        == "revert_or_tool"
    )
    assert (
        filter_contrib(make_contrib(tags=frozenset({"AWB"}))).reason == "revert_or_tool"
    )
    assert (
        filter_contrib(make_contrib(tags=frozenset({"contenttranslation"}))).reason
        == "revert_or_tool"
    )


def test_user_warning_summaries_are_automated():
    warning = filter_contrib(
        make_contrib(
            ns=3,
            title="User talk:Example",
            sizediff=400,
            comment="Warning (level 1) ([[WP:TW|TW]])",
        )
    )
    assert warning.reason == "automated"
    assert (
        filter_contrib(make_contrib(tags=frozenset({"RedWarn"}))).reason == "automated"
    )
    assert (
        filter_contrib(make_contrib(tags=frozenset({"Ultraviolet"}))).reason
        == "automated"
    )


def test_wikilove_only_on_user_talk():
    comment = "A new WikiLove message for you!"
    assert filter_contrib(make_contrib(ns=0, comment=comment)).accept
    decision = filter_contrib(make_contrib(ns=3, sizediff=200, comment=comment))
    assert decision.reason == "automated"


def test_visualeditor_and_reverted_are_kept():
    assert filter_contrib(make_contrib(tags=frozenset({"visualeditor"}))).accept
    assert filter_contrib(make_contrib(tags=frozenset({"mw-reverted"}))).accept
    assert filter_contrib(make_contrib(tags=frozenset({"discussiontools"}))).accept


def test_excluded_namespace():
    decision = filter_contrib(make_contrib(ns=2, sizediff=10))
    assert decision.reason == "excluded_namespace"


def test_moved_from_creation():
    decision = filter_contrib(
        make_contrib(is_new=True, parentid=0, comment="moved from [[Old title]]")
    )
    assert decision.reason == "copied_or_restored"


def test_texthidden():
    result = process_edit(make_contrib(), None, "Some new text that is long enough.")
    assert result.reject_reason == "texthidden"
    assert result.passages == ()
