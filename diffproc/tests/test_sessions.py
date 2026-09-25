from datetime import timedelta

from diffproc.config import PipelineConfig
from diffproc.sessions import collapse_session, iter_sessions
from diffproc.tests.fixtures import make_contrib


def _edit(revid: int, parentid: int, minute: int, sizediff: int = 40, **overrides):
    return make_contrib(
        revid=revid,
        parentid=parentid,
        sizediff=sizediff,
        timestamp=f"2026-06-15T12:{minute:02d}:00Z",
        **overrides,
    )


def test_small_saves_on_one_page_collapse_to_the_net_change():
    edits = [
        _edit(30, 20, 30),
        _edit(20, 10, 20),
        _edit(10, 5, 10),
    ]
    sessions = list(iter_sessions(edits, PipelineConfig()))
    assert len(sessions) == 1
    net = collapse_session(sessions[0])
    assert net.revid == 30
    assert net.parentid == 5
    assert net.sizediff == 120


def test_a_gap_over_an_hour_starts_a_new_run():
    edits = [
        _edit(30, 20, 30),
        make_contrib(
            revid=20,
            parentid=10,
            sizediff=40,
            timestamp="2026-06-15T11:00:00Z",
        ),
    ]
    sessions = list(iter_sessions(edits, PipelineConfig(), gap=timedelta(hours=1)))
    assert [collapse_session(session).revid for session in sessions] == [30, 20]


def test_another_page_does_not_split_the_run():
    edits = [
        _edit(30, 20, 40),
        make_contrib(revid=99, parentid=98, pageid=7, sizediff=40, timestamp="2026-06-15T12:30:00Z"),
        _edit(20, 10, 20),
    ]
    sessions = list(iter_sessions(edits, PipelineConfig()))
    page_runs = [session for session in sessions if session[0].pageid == 100]
    assert len(page_runs) == 1
    assert collapse_session(page_runs[0]).parentid == 10


def test_another_editor_in_between_splits_the_run():
    edits = [
        _edit(30, 25, 30),
        _edit(20, 10, 20),
    ]
    sessions = list(iter_sessions(edits, PipelineConfig()))
    assert [collapse_session(session).revid for session in sessions] == [30, 20]


def test_a_rollback_between_saves_splits_the_run():
    edits = [
        _edit(30, 25, 30),
        _edit(25, 20, 25, tags=frozenset({"mw-rollback"})),
        _edit(20, 10, 20),
    ]
    sessions = list(iter_sessions(edits, PipelineConfig()))
    assert [collapse_session(session).revid for session in sessions] == [30, 20]
