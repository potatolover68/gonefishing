import os
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("PYWIKIBOT_NO_USER_CONFIG", "2")

import pywikibot
import zstandard
from pywikibot.login import ClientLoginManager

from diffproc.config import PipelineConfig
from diffproc.types import Contrib

_COMPRESSOR = zstandard.ZstdCompressor(level=3)
_DECOMPRESSOR = zstandard.ZstdDecompressor()
_CONTRIB_PROPS = "ids|title|timestamp|comment|sizediff|flags|tags"


class RevisionCache:
    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._local = threading.local()
        connection = self._connection()
        connection.execute(
            "CREATE TABLE IF NOT EXISTS revisions ("
            "revid INTEGER PRIMARY KEY, content BLOB, hidden INTEGER NOT NULL)"
        )
        connection.commit()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._path, check_same_thread=False)
            self._local.connection = connection
        return connection

    def get(self, revid: int) -> tuple[str | None, bool] | None:
        row = self._connection().execute(
            "SELECT content, hidden FROM revisions WHERE revid = ?",
            (revid,),
        ).fetchone()
        if row is None:
            return None
        blob, hidden = row
        if hidden:
            return None, True
        return _DECOMPRESSOR.decompress(blob).decode("utf-8"), False

    def put(self, revid: int, content: str | None, hidden: bool) -> None:
        blob = (
            None if content is None else _COMPRESSOR.compress(content.encode("utf-8"))
        )
        connection = self._connection()
        connection.execute(
            "INSERT OR REPLACE INTO revisions (revid, content, hidden) VALUES (?, ?, ?)",
            (revid, blob, 1 if hidden else 0),
        )
        connection.commit()

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            return
        connection.close()
        self._local.connection = None


def _revision_parts(rev: dict) -> tuple[int, str | None, bool]:
    revid = int(rev["revid"])
    slot = rev.get("slots", {}).get("main", {})
    if not isinstance(slot, dict):
        slot = {}
    if "texthidden" in slot or "texthidden" in rev:
        return revid, None, True
    for source in (slot, rev):
        if "content" in source:
            return revid, source["content"], False
        if "*" in source:
            return revid, source["*"], False
    return revid, None, True


def _iter_revisions(payload: dict) -> Iterator[dict]:
    pages = payload.get("query", {}).get("pages", {})
    values = pages.values() if isinstance(pages, dict) else pages
    for page in values:
        for rev in page.get("revisions") or []:
            yield rev


def _configure(config: PipelineConfig) -> None:
    pywikibot.config.user_agent_format = config.user_agent
    pywikibot.config.maxlag = config.maxlag


def _account_name(username: str) -> str:
    return username.split("@", 1)[0]


def _active_account(site) -> str | None:
    info = getattr(site, "userinfo", None)
    if not isinstance(info, dict) or "anon" in info or not info.get("id"):
        return None
    name = info.get("name")
    return name if isinstance(name, str) else None


def _is_wiki_cookie(domain: str, host: str, family_domain: str) -> bool:
    bare = domain.lstrip(".")
    return bare == host or bare == family_domain or host.endswith("." + bare)


def _drop_session_cookies(site) -> None:
    """Drop a bot-password session so action=login is allowed.

    MediaWiki rejects action=login when the request already carries a
    BotPasswordSessionProvider cookie such as enwiki_BPsession.
    """
    from pywikibot.comms import http

    host = site.hostname()
    family_domain = site.family.domain
    for cookie in list(http.cookie_jar):
        if _is_wiki_cookie(cookie.domain, host, family_domain):
            http.cookie_jar.clear(cookie.domain, cookie.path, cookie.name)
    site.tokens.clear()
    del site.userinfo


class WikiClient:
    def __init__(self, config: PipelineConfig, cache: RevisionCache, site=None) -> None:
        if not config.user_agent.strip():
            raise ValueError("user_agent is required")
        self._config = config
        self._cache = cache
        _configure(config)
        self.site = site if site is not None else pywikibot.Site("en", "wikipedia")

    def close(self) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        active = _active_account(self.site)
        if active is not None and active.casefold() == _account_name(username).casefold():
            return
        _drop_session_cookies(self.site)
        manager = ClientLoginManager(password=password, site=self.site, user=username)
        if not manager.login():
            raise RuntimeError("login failed")

    def recent_contribs(self, user: str, namespaces: list[int], limit: int = 2500) -> list[Contrib]:
        rows = self.site.usercontribs(
            user=user,
            namespaces=namespaces,
            prop=_CONTRIB_PROPS,
            total=limit,
        )
        return [_contrib_from_usercontribs(row, user) for row in rows]

    def iter_user_contribs(self, user: str, namespaces: list[int]) -> Iterator[Contrib]:
        rows = self.site.usercontribs(
            user=user,
            namespaces=namespaces,
            prop=_CONTRIB_PROPS,
        )
        for row in rows:
            yield _contrib_from_usercontribs(row, user)

    def _submit(self, **params) -> dict:
        return self.site.simple_request(**params).submit()

    def local_groups(self, username: str) -> set[str]:
        payload = self._submit(action="query", list="users", ususers=username, usprop="groups")
        users = payload.get("query", {}).get("users") or []
        if not users:
            return set()
        groups = users[0].get("groups") or []
        return {str(group) for group in groups}

    def global_groups(self, username: str) -> set[str]:
        payload = self._submit(
            action="query",
            meta="globaluserinfo",
            guiuser=username,
            guiprop="groups",
        )
        info = payload.get("query", {}).get("globaluserinfo") or {}
        groups = info.get("groups") or []
        if groups and isinstance(groups[0], dict):
            return {str(group.get("name") or "") for group in groups}
        return {str(group) for group in groups}

    def user_facts(self, names: list[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        batch = self._config.request_limit()
        for start in range(0, len(names), batch):
            chunk = [name for name in names[start : start + batch] if "|" not in name]
            if not chunk:
                continue
            payload = self._submit(
                action="query",
                list="users",
                ususers="|".join(chunk),
                usprop="editcount|registration|blockinfo",
            )
            for user in payload.get("query", {}).get("users") or []:
                if user.get("missing") or user.get("invalid") or not user.get("name"):
                    continue
                found[str(user["name"])] = user
        return found

    def fetch_revisions(self, revids: list[int]) -> dict[int, str | None]:
        found: dict[int, str | None] = {}
        missing: list[int] = []
        for revid in revids:
            cached = self._cache.get(revid)
            if cached is None:
                missing.append(revid)
                continue
            content, hidden = cached
            found[revid] = None if hidden else content
        batch = self._config.request_limit()
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            payload = self._submit(
                action="query",
                prop="revisions",
                revids="|".join(str(revid) for revid in chunk),
                rvprop="ids|content",
                rvslots="main",
            )
            for rev in _iter_revisions(payload):
                revid, content, hidden = _revision_parts(rev)
                self._cache.put(revid, None if hidden else content, hidden)
                found[revid] = None if hidden else content
        return found


def _contrib_from_usercontribs(row: dict, user: str) -> Contrib:
    return Contrib(
        revid=int(row["revid"]),
        parentid=int(row.get("parentid") or 0),
        pageid=int(row.get("pageid") or 0),
        ns=int(row.get("ns") or 0),
        title=row.get("title") or "",
        timestamp=row.get("timestamp") or "",
        comment=row.get("comment") or "",
        sizediff=int(row.get("sizediff") or 0),
        tags=frozenset(row.get("tags") or []),
        is_new=bool(row.get("new")),
        user=user,
    )
