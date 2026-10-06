from datetime import datetime, timedelta, timezone

import numpy as np

from authorship.lookup import centroid_distances, tag_similarity

from identity import experienced_editor, normalize_username


def test_tag_similarity_drops_the_excluded_member():
    query = np.array([1.0, 0.0], dtype=np.float32)
    same = np.array([2.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 2.0], dtype=np.float32)
    assert tag_similarity(query, [other], [2]) == 0
    assert tag_similarity(query, [same, other], [2, 2]) > 0.5


def test_adding_accounts_to_a_tag_is_queued(tmp_path, monkeypatch):
    from factory import create_app
    from models import AppUser, LookupJob, db

    monkeypatch.setenv("GONEFISHING_NO_JOBS", "1")
    monkeypatch.setattr("factory._forget_tag_scores", lambda name: None)
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

    first = client.post(
        "/tags",
        data={"csrf": "token", "tag": "Expertwikiguy", "account": ["Lionsonny"]},
    )
    assert first.status_code == 302
    second = client.post(
        "/tags",
        data={"csrf": "token", "tag": "Expertwikiguy", "account": ["Darrenchant"]},
    )
    assert second.status_code == 302
    with app.app_context():
        jobs = LookupJob.query.filter_by(kind="tag").all()
        assert len(jobs) == 1
        assert jobs[0].tag_name == "Expertwikiguy"
        assert jobs[0].status == "queued"
    page = client.get("/")
    assert b"Expertwikiguy" in page.data
    assert b"tag update - queued" in page.data


def test_centroid_distance_is_zero_for_the_same_vector():
    same = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    scores = centroid_distances({"same": same, "other": other})
    assert scores["same"] == scores["other"]
    assert scores["same"] > 0
    alone = centroid_distances({"same": same, "copy": same})
    assert alone["same"] == 1.0
    assert alone["copy"] == 1.0


def test_experienced_editor_needs_many_edits_and_six_months():
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    assert experienced_editor(1001, "2026-04-03T00:00:00Z", now)
    assert not experienced_editor(1000, "2020-01-01T00:00:00Z", now)
    assert not experienced_editor(5000, "2026-04-07T00:00:00Z", now)
    assert not experienced_editor(5000, "", now)


def test_pages_require_login(tmp_path):
    from factory import create_app

    app = create_app(init_heavy=False, db_path=tmp_path / "app.sqlite")
    client = app.test_client()
    for path in ("/", "/log", "/tag/Expertwikiguy", "/tags"):
        response = client.get(path)
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/login")


def test_tag_add_remove_and_undo(tmp_path, monkeypatch):
    from factory import create_app
    from models import AppUser, LogEntry, LookupJob, Tag, TagMember, db

    monkeypatch.setenv("GONEFISHING_NO_JOBS", "1")
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

    queued = client.post("/lookup", data={"csrf": "token", "username": "Lionsonny"})
    assert queued.status_code == 302
    again = client.post("/lookup", data={"csrf": "token", "username": "Darrenchant"})
    assert again.status_code == 302
    with app.app_context():
        names = [job.wiki_username for job in LookupJob.query.order_by(LookupJob.id)]
        assert names == ["Lionsonny", "Darrenchant"]

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


def test_cpu_quota_rounds_to_cores():
    from cpus import cpus_from_quota

    assert cpus_from_quota("max 100000") is None
    assert cpus_from_quota("200000 100000") == 2
    assert cpus_from_quota("50000 100000") == 1


def test_one_core_stays_free_for_the_site():
    from cpus import compute_cpus, provisioned_cpus

    assert compute_cpus() == max(1, provisioned_cpus() - 1)


def test_local_lookup_hands_diffs_to_the_client(tmp_path, monkeypatch):
    from factory import create_app
    from models import AppUser, LogEntry, LookupJob, db

    monkeypatch.setenv("GONEFISHING_NO_JOBS", "1")
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

    started = client.post(
        "/lookup",
        data={"csrf": "token", "username": "Ada", "local": "on"},
    )
    assert started.status_code == 200
    qid = started.get_json()["qid"]
    again = client.post(
        "/lookup",
        data={"csrf": "token", "username": "Ada", "local": "on"},
    )
    assert again.get_json()["qid"] == qid
    assert client.get(f"/local?qid={qid}").status_code == 202

    with app.app_context():
        job = LookupJob.query.filter_by(public_id=qid).one()
        assert job.local is True
        job.status = "ready"
        job.useful_count = 2
        db.session.commit()

    class Store:
        def __init__(self):
            self.saved = None

        def prose(self, user):
            return ["alpha", "beta"]

        def put_vector(self, user, vector, processor=""):
            self.saved = (user, processor, int(vector.shape[0]))

    store = Store()
    monkeypatch.setattr("factory._services", lambda: {"store": store})
    monkeypatch.setattr("factory._comparison", lambda *args, **kwargs: [])

    ready = client.get(f"/local?qid={qid}")
    assert ready.status_code == 200
    assert ready.get_json()["diffs"] == ["alpha", "beta"]
    waiting = client.get("/?user=Ada")
    assert b"Waiting for a local embedding of Ada." in waiting.data

    posted = client.post(
        f"/local?qid={qid}",
        json=[0.25] * 512,
        headers={"X-CSRF-Token": "token"},
    )
    assert posted.status_code == 200
    assert store.saved == ("Ada", "MSK", 512)
    with app.app_context():
        assert LookupJob.query.filter_by(public_id=qid).one().status == "done"
        assert LogEntry.query.filter_by(action="lookup", lookup_name="Ada").count() == 1
    assert b"(local)" in client.get("/").data


def test_stale_local_jobs_are_not_done(tmp_path, monkeypatch):
    from factory import create_app
    from models import AppUser, LookupJob, db

    monkeypatch.setenv("GONEFISHING_NO_JOBS", "1")
    app = create_app(init_heavy=False, db_path=tmp_path / "app.sqlite")
    with app.app_context():
        actor = AppUser(username="MSK")
        db.session.add(actor)
        db.session.commit()
        actor_id = actor.id
        db.session.add(
            LookupJob(
                actor_id=actor_id,
                wiki_username="Ada",
                local=True,
                public_id="stale-qid",
                status="ready",
                created_at=datetime.now(timezone.utc) - timedelta(minutes=11),
            )
        )
        db.session.commit()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = actor_id
        sess["username"] = "MSK"
        sess["csrf"] = "token"

    stalled = client.get("/local?qid=stale-qid")
    assert stalled.status_code == 422
    assert stalled.get_json()["error"] == "not done"
    with app.app_context():
        assert (
            LookupJob.query.filter_by(public_id="stale-qid").one().status == "not done"
        )
    assert b"not done" in client.get("/").data


def test_normalize_username():
    assert normalize_username(" User_Tamzin ") == "User Tamzin"
    assert normalize_username(" User_Tamzin ") == "User Tamzin"
