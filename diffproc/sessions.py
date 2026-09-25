"""Group an uninterrupted run of saves on one page into a single edit."""

from collections.abc import Iterator
from datetime import datetime, timedelta

from diffproc.config import PipelineConfig
from diffproc.contribs import filter_contrib
from diffproc.namespaces import ns_group
from diffproc.types import Contrib

SESSION_GAP = timedelta(hours=1)


def _parsed(timestamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def extends_session(newer: Contrib, older: Contrib, gap: timedelta = SESSION_GAP) -> bool:
    """True when older is the previous save in newer's run.

    newer and older are consecutive revisions by the same account on one page,
    and the later save happened within gap of the earlier one.
    """
    if newer.pageid == 0 or newer.pageid != older.pageid:
        return False
    if newer.parentid != older.revid:
        return False
    started = _parsed(older.timestamp)
    finished = _parsed(newer.timestamp)
    if started is None or finished is None:
        return False
    return finished - started <= gap


def collapse_session(edits: list[Contrib]) -> Contrib:
    """Represent a newest-first run by its net change."""
    newest, oldest = edits[0], edits[-1]
    return Contrib(
        revid=newest.revid,
        parentid=oldest.parentid,
        pageid=newest.pageid,
        ns=newest.ns,
        title=newest.title,
        timestamp=newest.timestamp,
        comment=newest.comment,
        sizediff=sum(edit.sizediff for edit in edits),
        tags=frozenset().union(*(edit.tags for edit in edits)),
        is_new=oldest.is_new,
        user=newest.user,
    )


def iter_sessions(
    contribs: Iterator[Contrib] | list[Contrib],
    config: PipelineConfig | None = None,
    gap: timedelta = SESSION_GAP,
) -> Iterator[list[Contrib]]:
    """Yield newest-first runs. contribs must also be newest-first."""
    config = config or PipelineConfig()
    open_sessions: dict[int, list[Contrib]] = {}
    for contrib in contribs:
        if not filter_contrib(contrib, config, check_size=False).accept:
            if contrib.pageid in open_sessions:
                yield open_sessions.pop(contrib.pageid)
            continue
        current = open_sessions.get(contrib.pageid)
        if current and extends_session(current[-1], contrib, gap):
            current.append(contrib)
            continue
        if current:
            yield current
        open_sessions[contrib.pageid] = [contrib]
    yield from open_sessions.values()


def session_is_large_enough(edits: list[Contrib], config: PipelineConfig) -> bool:
    group = ns_group(edits[0].ns, config.content_ids)
    if group is None:
        return False
    return sum(edit.sizediff for edit in edits) >= config.thresholds_for(group).min_sizediff
