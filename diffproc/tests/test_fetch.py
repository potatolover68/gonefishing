import threading

import pytest

from diffproc.config import PipelineConfig
from diffproc.fetch import RevisionCache, WikiClient, _is_wiki_cookie


def _config() -> PipelineConfig:
    return PipelineConfig(user_agent="gonefishing-tests/0.1 (offline)")


def test_bot_request_limit_is_500():
    assert PipelineConfig().request_limit() == 50
    assert PipelineConfig(is_bot=True).request_limit() == 500


def _pages(revisions: list[dict]) -> dict:
    return {"query": {"pages": {"1": {"revisions": revisions}}}}


class _Request:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def submit(self) -> dict:
        return self._payload


class _Site:
    def __init__(self, handler) -> None:
        self.calls = 0
        self.params: list[dict] = []
        self._handler = handler

    def simple_request(self, **params) -> _Request:
        self.calls += 1
        self.params.append(params)
        return _Request(self._handler(params))


def test_wiki_cookie_matches_bot_password_session():
    assert _is_wiki_cookie("en.wikipedia.org", "en.wikipedia.org", "wikipedia.org")
    assert _is_wiki_cookie(".wikipedia.org", "en.wikipedia.org", "wikipedia.org")
    assert not _is_wiki_cookie("commons.wikimedia.org", "en.wikipedia.org", "wikipedia.org")


def test_login_reuses_bot_password_session(tmp_path, monkeypatch):
    cache = RevisionCache(tmp_path / "revs.sqlite")
    site = _Site(lambda _params: {})
    site.userinfo = {"id": 1, "name": "MSK"}
    client = WikiClient(_config(), cache, site=site)
    monkeypatch.setattr(
        "diffproc.fetch.ClientLoginManager",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("login was attempted")),
    )
    client.login("MSK@diffproc", "secret")
    client.close()
    cache.close()


def test_login_drops_bot_password_cookie(tmp_path, monkeypatch):
    from http.cookiejar import Cookie

    from pywikibot.comms import http

    cookie = Cookie(
        version=0,
        name="enwiki_BPsession",
        value="stale",
        port=None,
        port_specified=False,
        domain="en.wikipedia.org",
        domain_specified=True,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=True,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={},
    )
    http.cookie_jar.set_cookie(cookie)

    class _Tokens:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class _Manager:
        def __init__(self, **_kwargs) -> None:
            names = [item.name for item in http.cookie_jar if item.domain == "en.wikipedia.org"]
            assert "enwiki_BPsession" not in names

        def login(self) -> bool:
            return True

    site = _Site(lambda _params: {})
    site.userinfo = {"anon": True, "id": 0, "name": "127.0.0.1"}
    site.hostname = lambda: "en.wikipedia.org"
    site.family = type("Family", (), {"domain": "wikipedia.org"})()
    site.tokens = _Tokens()
    monkeypatch.setattr("diffproc.fetch.ClientLoginManager", _Manager)
    cache = RevisionCache(tmp_path / "revs.sqlite")
    client = WikiClient(_config(), cache, site=site)
    client.login("MSK@diffproc", "secret")
    assert site.tokens.cleared
    assert not hasattr(site, "userinfo")
    client.close()
    cache.close()


def test_cache_can_be_read_from_another_thread(tmp_path):
    cache = RevisionCache(tmp_path / "revs.sqlite")
    cache.put(1, "Hello there.", False)
    found: dict[int, tuple[str | None, bool] | None] = {}

    def read() -> None:
        found[1] = cache.get(1)

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    assert found[1] == ("Hello there.", False)
    cache.close()


def test_user_agent_required(tmp_path):
    cache = RevisionCache(tmp_path / "revs.sqlite")
    with pytest.raises(ValueError):
        WikiClient(PipelineConfig(), cache)
    cache.close()


def test_cache_hit_skips_second_request(tmp_path):
    site = _Site(
        lambda _params: _pages(
            [{"revid": 5, "slots": {"main": {"content": "Hello there."}}}]
        )
    )
    cache = RevisionCache(tmp_path / "revs.sqlite")
    client = WikiClient(_config(), cache, site=site)
    assert client.fetch_revisions([5]) == {5: "Hello there."}
    assert client.fetch_revisions([5]) == {5: "Hello there."}
    assert site.calls == 1
    client.close()
    cache.close()


def test_hidden_revision_is_cached(tmp_path):
    site = _Site(
        lambda _params: _pages(
            [{"revid": 7, "slots": {"main": {"texthidden": True}}}]
        )
    )
    cache = RevisionCache(tmp_path / "revs.sqlite")
    client = WikiClient(_config(), cache, site=site)
    assert client.fetch_revisions([7]) == {7: None}
    assert client.fetch_revisions([7]) == {7: None}
    assert site.calls == 1
    client.close()
    cache.close()
