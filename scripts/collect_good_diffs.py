"""Collect useful diffs per account."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffproc.config import PipelineConfig
from diffproc.fetch import WikiClient
from diffproc.namespaces import groups_from_siteinfo, ns_group
from diffproc.pipeline import process_edit
from diffproc.sessions import collapse_session, iter_sessions, session_is_large_enough

TARGET = 250
MAX_CONTRIBS = 2500


def load_env(path: Path | None = None) -> dict[str, str]:
    """Load .env without overriding variables already set in the environment."""
    if path is None or path.exists():
        load_dotenv(path, override=False)
    return dict(os.environ)


def included_namespaces(client: WikiClient) -> list[int]:
    namespaces = client.site.siteinfo["namespaces"]
    content_ids = groups_from_siteinfo(namespaces)
    chosen: list[int] = []
    for key, info in namespaces.items():
        ns = int(info.get("id", key))
        if ns_group(ns, content_ids) is not None:
            chosen.append(ns)
    return sorted(set(chosen))


def record(contrib, result) -> dict:
    passages = [passage for passage in result.passages if passage.kind == "new"]
    return {
        "user": contrib.user,
        "revid": contrib.revid,
        "parentid": contrib.parentid,
        "pageid": contrib.pageid,
        "ns": contrib.ns,
        "ns_group": passages[0].ns_group if passages else result.passages[0].ns_group,
        "title": contrib.title,
        "timestamp": contrib.timestamp,
        "passages": [
            {
                "prose": passage.prose,
                "raw_wikitext": passage.raw_wikitext,
                "n_sentences": passage.n_sentences,
                "n_chars": passage.n_chars,
                "inserted_chars_total": passage.inserted_chars_total,
                "kind": passage.kind,
            }
            for passage in passages
        ],
    }


def collect_user(
    client: WikiClient,
    user: str,
    namespaces: list[int],
    config: PipelineConfig,
    dest: Path,
    max_useful: int = TARGET,
    max_scanned: int = MAX_CONTRIBS,
    facts: list | None = None,
) -> int:
    kept = 0
    scanned = 0
    batch: list = []

    def flush() -> None:
        nonlocal kept
        if not batch or kept >= max_useful:
            batch.clear()
            return
        revids: list[int] = []
        for contrib in batch:
            revids.append(contrib.revid)
            if contrib.parentid and not contrib.is_new:
                revids.append(contrib.parentid)
        texts = client.fetch_revisions(revids)
        with dest.open("a", encoding="utf-8") as handle:
            for contrib in batch:
                if kept >= max_useful:
                    break
                new_text = texts.get(contrib.revid)
                if contrib.is_new:
                    old_text = ""
                else:
                    old_text = texts.get(contrib.parentid)
                if new_text is None or old_text is None:
                    continue
                result = process_edit(contrib, old_text, new_text, config)
                if not result.useful:
                    continue
                handle.write(
                    json.dumps(record(contrib, result), ensure_ascii=False) + "\n"
                )
                kept += 1
        batch.clear()

    def limited():
        nonlocal scanned
        for contrib in client.iter_user_contribs(user, namespaces):
            scanned += 1
            if scanned > max_scanned or kept >= max_useful:
                return
            if facts is not None:
                facts.append(contrib)
            yield contrib

    for session in iter_sessions(limited(), config):
        if not session_is_large_enough(session, config):
            continue
        batch.append(collapse_session(session))
        if len(batch) >= 25:
            flush()
    flush()
    return kept