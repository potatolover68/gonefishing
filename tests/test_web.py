from datetime import datetime, timezone

from identity import (
    account_stats,
    age_phrase,
    clerk_keys,
    named_allowed,
    normalize_username,
)


def test_allowlist_is_named_allowed(tmp_path):
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("Tamzin\nLuniZunie \n", encoding="utf-8")
    allowed = clerk_keys(allowlist)
    assert named_allowed("MSK", allowed)
    assert named_allowed("msk", allowed)
    assert named_allowed("_Tamzin", allowed)
    assert named_allowed("LuniZunie", allowed)
    assert not named_allowed("SomeoneElse", allowed)


def test_account_stats_include_age_and_useful_count():
    now = datetime(2026, 3, 5, tzinfo=timezone.utc)
    registered = datetime(2019, 3, 5, tzinfo=timezone.utc)
    assert age_phrase(registered, now) == "7 years"
    assert (
        account_stats(15234, "2019-03-05T00:00:00Z", 42, now)
        == "15234 (5 March 2019, 7 years, 42 useful)"
    )


def test_pages_require_login(tmp_path):
    from factory import create_app

    app = create_app(init_heavy=False, db_path=tmp_path / "app.sqlite")
    client = app.test_client()
    for path in ("/", "/log", "/tag/Expertwikiguy", "/tags"):
        response = client.get(path)
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/login")


def test_tag_add_remove_and_undo(tmp_path):
    from factory import create_app
    from models import AppUser, LogEntry, Tag, TagMember, db

    app = create_app(init_heavy=False, db_path=tmp_path / "app.sqlite")
    with app.app_context():
        actor = AppUser(username="MSK")
        db.session.add(actor)
        db.session.commit()
        actor_id = actor.id
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = actor_id
        sess["username"] = "MSK"
        sess["csrf"] = "token"

    home = client.get("/")
    assert home.status_code == 200
    assert b"Wikipedia username" in home.data

    added = client.post(
        "/tags",
        data={
            "csrf": "token",
            "tag": "Expertwikiguy",
            "account": ["Lionsonny", "Darrenchant"],
        },
    )
    assert added.status_code == 302
    with app.app_context():
        tag = Tag.query.filter_by(name="Expertwikiguy").one()
        assert {
            member.wiki_username for member in TagMember.query.filter_by(tag_id=tag.id)
        } == {
            "Lionsonny",
            "Darrenchant",
        }
        entry = LogEntry.query.filter_by(action="tag_add").one()
        entry_id = entry.id
        assert entry.names() == ["Lionsonny", "Darrenchant"]

    page = client.get("/tag/Expertwikiguy")
    assert page.status_code == 200
    assert b"Lionsonny" in page.data
    assert b"Added by MSK" in page.data

    removed = client.post(
        "/tag/Expertwikiguy/remove",
        data={"csrf": "token", "account": "Lionsonny"},
    )
    assert removed.status_code == 302
    undo_remove = client.get("/log")
    assert b"removed" in undo_remove.data
    with app.app_context():
        remove_entry = LogEntry.query.filter_by(action="tag_remove").one()
        remove_id = remove_entry.id

    restored = client.post(f"/log/{remove_id}/undo", data={"csrf": "token"})
    assert restored.status_code == 302
    with app.app_context():
        names = {member.wiki_username for member in TagMember.query.all()}
        assert names == {"Lionsonny", "Darrenchant"}
        adder = (
            TagMember.query.filter_by(wiki_username="Lionsonny").one().added_by.username
        )
        assert adder == "MSK"
        assert LogEntry.query.filter_by(action="undo_remove").count() == 1
        assert db.session.get(LogEntry, remove_id).undone is True

    client.post(f"/log/{entry_id}/undo", data={"csrf": "token"})
    with app.app_context():
        assert TagMember.query.count() == 0
        assert db.session.get(LogEntry, entry_id).undone is True
        undo_add = LogEntry.query.filter_by(action="undo_add").one()
        undo_add_id = undo_add.id
    client.post(f"/log/{undo_add_id}/undo", data={"csrf": "token"})
    with app.app_context():
        assert {member.wiki_username for member in TagMember.query.all()} == {
            "Lionsonny",
            "Darrenchant",
        }
        assert db.session.get(LogEntry, undo_add_id).undone is True


def test_notes_empty_tag_and_tag_index(tmp_path):
    from factory import create_app
    from models import AppUser, LogEntry, Tag, TagMember, db

    app = create_app(init_heavy=False, db_path=tmp_path / "app.sqlite")
    with app.app_context():
        actor = AppUser(username="MSK")
        db.session.add(actor)
        db.session.commit()
        actor_id = actor.id
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = actor_id
        sess["username"] = "MSK"
        sess["csrf"] = "token"

    client.post("/tags", data={"csrf": "token", "tag": "Notesy", "account": "Ada"})
    page = client.get("/tag/Notesy")
    assert b'<p id="tag-notes"' in page.data
    assert b'id="notes-form"' in page.data
    assert b"hidden" in page.data
    saved = client.post(
        "/tag/Notesy/notes", data={"csrf": "token", "notes": "Watch this cluster."}
    )
    assert saved.status_code == 302
    shown = client.get("/tag/Notesy")
    assert b"Watch this cluster." in shown.data
    listing = client.get("/tags")
    assert b"Notesy" in listing.data
    blocked = client.post("/tag/Notesy/delete", data={"csrf": "token"})
    assert blocked.status_code == 400
    with app.app_context():
        note = LogEntry.query.filter_by(action="note_edit").one()
        note_id = note.id
        assert "UTC" in note.stamp()
    diff = client.get(f"/log/{note_id}/diff")
    assert b"Watch this cluster." in diff.data
    log = client.get("/log")
    assert b"diff" in log.data
    assert b"UTC" in log.data
    client.post("/tag/Notesy/remove", data={"csrf": "token", "account": "Ada"})
    deleted = client.post("/tag/Notesy/delete", data={"csrf": "token"})
    assert deleted.status_code == 302
    assert client.get("/tag/Notesy").status_code == 404
    with app.app_context():
        assert Tag.query.filter_by(name="Notesy").one_or_none() is None
        assert TagMember.query.count() == 0
        assert LogEntry.query.filter_by(action="tag_delete").count() == 1


def test_on_tf_keeps_the_database_under_home(tmp_path, monkeypatch):
    from factory import create_app

    home = tmp_path / "toolhome"
    monkeypatch.setenv("ON_TF", "1")
    monkeypatch.setenv("HOME", str(home))
    repo = tmp_path / "repo"
    (repo / "templates").mkdir(parents=True)
    (repo / "data").mkdir()
    app = create_app(init_heavy=False, root=repo)
    database = home / "gonefishing" / "app.sqlite"
    assert database.is_file()
    assert database.as_posix() in app.config["SQLALCHEMY_DATABASE_URI"]


def test_normalize_username():
    assert normalize_username(" User_Tamzin ") == "User Tamzin"
