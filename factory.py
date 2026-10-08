"""Flask lookup app. Wikimedia login is required for every page except login."""

import json
import os
import secrets
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
from authlib.integrations.base_client import OAuthError
from authlib.integrations.flask_client import OAuth
from flask import (
    Flask,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from authorship.corpus import Document, load_documents
from authorship.encode import LuarEncoder, luar_onnx_path
from authorship.lookup import AuthorStore, centroid_distances, tag_similarity
from authorship.signals import (
    blend_similarity,
    collect_topic_vector,
    cosine_dicts,
    cosine_histograms,
    fact_from_contrib,
    hour_histogram,
)
from diffproc.config import GroupThresholds, PipelineConfig
from diffproc.fetch import RevisionCache, WikiClient
from identity import (
    account_stats,
    block_text,
    clerk_keys,
    experienced_editor,
    name_key,
    named_allowed,
    normalize_username,
    parse_timestamp,
)
from models import AppUser, LogEntry, LookupJob, Tag, TagMember, WikiProfile, db
from scripts.collect_good_diffs import collect_user, included_namespaces, load_env

ROOT = Path(__file__).resolve().parent
MAX_USEFUL = 500
VECTOR_DIM = 512
LOCAL_JOB_TTL = timedelta(minutes=10)
EXPERIENCED_NOTICE = (
    "<b>Note:</b> by its very nature, good encyclopedic writing is dispassionate and bland, "
    "so when comparing experienced editors take the similarity score with an "
    "extra large grain of salt."
)
OPEN_ENDPOINTS = {"login", "login_start", "callback"}


@dataclass
class ViewRow:
    kind: str
    name: str
    similarity: float | None = None
    stats: str = ""
    block: str = ""
    edits: int | None = None
    created: str = ""
    blocked: bool = False
    tags: list[tuple[str, str]] = field(default_factory=list)
    talk: str = ""
    contribs: str = ""
    centralauth: str = ""
    blocklog: str = ""
    pinned: bool = False
    local_by: str = ""
    similarity_detail: str = ""


def wiki_slug(name: str) -> str:
    return quote(name.replace(" ", "_"), safe="")


def account_links(name: str) -> dict[str, str]:
    slug = wiki_slug(name)
    return {
        "talk": f"https://en.wikipedia.org/wiki/User_talk:{slug}",
        "contribs": f"https://en.wikipedia.org/wiki/Special:Contributions/{slug}",
        "centralauth": f"https://meta.wikimedia.org/wiki/Special:CentralAuth/{slug}",
        "blocklog": f"https://en.wikipedia.org/wiki/Special:Log/block?page=User:{slug}",
    }


def runtime_data(root: Path) -> Path:

    if os.environ.get("ON_TF"):
        home = os.environ.get("TOOL_DATA_DIR") or os.environ.get("HOME")
        if not home:
            raise RuntimeError("ON_TF is set but HOME is not")
        data = Path(home) / "gonefishing"
    else:
        data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data


def _allowlist_file(data: Path, root: Path) -> Path:
    dest = data / "allowlist.txt"
    bundled = root / "data" / "allowlist.txt"
    if dest.resolve() != bundled.resolve() and not dest.exists() and bundled.is_file():
        dest.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def create_app(
    init_heavy: bool | None = None,
    root: Path | None = None,
    db_path: Path | None = None,
) -> Flask:
    root = root or ROOT
    if init_heavy is None:
        init_heavy = True
    app = Flask(__name__, template_folder=str(root / "templates"))
    env = load_env(root / ".env")
    data = runtime_data(root)
    database = (db_path or (data / "app.sqlite")).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + database.as_posix()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = (
        env.get("SECRET_KEY") or env.get("OAUTH_SECRET") or "gonefishing-dev"
    )
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    db.init_app(app)
    oauth = OAuth(app)
    oauth.register(
        name="wikimedia",
        client_id=env.get("OAUTH_KEY") or "missing",
        client_secret=env.get("OAUTH_SECRET") or "missing",
        access_token_url="https://meta.wikimedia.org/w/rest.php/oauth2/access_token",
        authorize_url="https://meta.wikimedia.org/w/rest.php/oauth2/authorize",
        api_base_url="https://meta.wikimedia.org/w/rest.php/oauth2/resource/",
        client_kwargs={"scope": "basic"},
    )
    app.extensions["oauth"] = oauth
    app.extensions["gone"] = {
        "env": env,
        "root": root,
        "allowlist": clerk_keys(_allowlist_file(data, root)),
        "client": None,
        "store": None,
        "encoder": None,
        "config": None,
        "namespaces": None,
        "data": data,
    }
    with app.app_context():
        db.create_all()
        _ensure_tag_notes()
        _ensure_lookup_jobs()
        if os.environ.get("GONEFISHING_NO_JOBS") != "1":
            _start_lookup_worker(app)
        if init_heavy:
            _load_services(app, env, root)

    @app.context_processor
    def inject():
        return {"csrf": csrf_token(), "me": session.get("username", "")}

    @app.before_request
    def guard():
        if request.method == "POST" and _supplied_csrf() != session.get("csrf"):
            abort(400)
        if request.endpoint in OPEN_ENDPOINTS:
            return None
        if session.get("user_id"):
            return None
        return redirect(url_for("login"))

    @app.get("/login")
    def login():
        if session.get("user_id"):
            return redirect(url_for("index"))
        return render_template(
            "login.html",
            denied=request.args.get("denied"),
            failed=request.args.get("failed"),
        )

    @app.get("/login/wikimedia")
    def login_start():
        return oauth.wikimedia.authorize_redirect()

    @app.get("/callback")
    def callback():
        try:
            token = oauth.wikimedia.authorize_access_token()
            profile = oauth.wikimedia.get("profile", token=token).json()
        except (OAuthError, OSError, ValueError):
            return redirect(url_for("login", failed=1))
        username = ""
        if isinstance(profile, dict):
            username = str(profile.get("username") or profile.get("name") or "")
        username = normalize_username(username)
        if not username or not _privileged(username):
            session.clear()
            return redirect(url_for("login", denied=1))
        actor = _actor(username)
        session.clear()
        session["user_id"] = actor.id
        session["username"] = actor.username
        session["csrf"] = secrets.token_urlsafe(32)
        return redirect(url_for("index"))

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    def index():
        _expire_local_jobs()
        subject = normalize_username(request.args.get("user", ""))
        rows: list[ViewRow] = []
        notice = ""
        active = None
        if subject:
            active = (
                LookupJob.query.filter(
                    LookupJob.wiki_username == subject,
                    LookupJob.status.in_(("queued", "running")),
                )
                .order_by(LookupJob.id.desc())
                .first()
            )
            if active is not None:
                running = LookupJob.query.filter_by(status="running").first()
                if active.status == "running":
                    notice = f"Comparing {subject}."
                elif running is not None and running.id != active.id:
                    notice = _busy_notice(running, subject)
                else:
                    notice = f"{subject} is waiting in the queue."
            elif _local_waiting(subject):
                notice = f"Waiting for a local embedding of {subject}."
            elif not _services()["store"].has_user(subject):
                notice = "No usable edits were stored for that account."
            else:
                rows = _comparison(subject)
                if not rows:
                    notice = "Nothing to compare."
                elif rows[0].pinned and experienced_editor(
                    rows[0].edits or 0,
                    rows[0].created,
                    datetime.now(timezone.utc),
                ):
                    notice = EXPERIENCED_NOTICE
        jobs = LookupJob.query.order_by(LookupJob.id.desc()).limit(20).all()
        tag_names = (
            [tag.name for tag in Tag.query.order_by(Tag.name).all()] if rows else []
        )
        return render_template(
            "index.html",
            subject=subject,
            rows=rows,
            notice=notice,
            jobs=jobs,
            tag_names=tag_names,
        )

    @app.post("/lookup")
    def lookup():
        username = normalize_username(request.form.get("username", ""))
        if not username:
            return redirect(url_for("index"))
        refresh = request.form.get("refresh") == "on"
        local = request.form.get("local", "") == "on"
        if local:
            inflight = (
                LookupJob.query.filter(
                    LookupJob.wiki_username == username,
                    LookupJob.local.is_(True),
                    LookupJob.status.in_(("queued", "running", "ready")),
                )
                .order_by(LookupJob.id.desc())
                .first()
            )
            if inflight is not None:
                return jsonify({"qid": inflight.public_id})
        existing = LookupJob.query.filter(
            LookupJob.wiki_username == username,
            LookupJob.status.in_(("queued", "running")),
        ).first()
        public_id = secrets.token_urlsafe(16) if local else ""
        if existing is None or local:
            db.session.add(
                LookupJob(
                    actor_id=_current_actor().id,
                    wiki_username=username,
                    refresh=refresh,
                    local=local,
                    public_id=public_id,
                    status="queued",
                    created_at=datetime.now(timezone.utc),
                )
            )
            db.session.commit()
        if local:
            return jsonify({"qid": public_id})
        return redirect(url_for("index", user=username))

    @app.get("/local")
    def local_diffs():
        job = _owned_local_job(request.args.get("qid", ""))
        if job.status in {"queued", "running"}:
            return jsonify({"status": job.status}), 202
        if job.status in {"error", "not done"}:
            return jsonify({"error": job.error or job.status}), 422
        texts = _services()["store"].prose(job.wiki_username)[:MAX_USEFUL]
        if not texts:
            return (
                jsonify({"error": "No usable edits were stored for that account."}),
                422,
            )
        return jsonify({"user": job.wiki_username, "diffs": texts})

    @app.post("/local")
    def submit_local():
        job = _owned_local_job(request.args.get("qid", ""))
        if job.status != "ready":
            abort(409)
        payload = request.get_json(silent=True)
        vector = _local_vector(payload)
        if vector is None:
            abort(400)
        actor = _current_actor()
        _services()["store"].put_vector(
            job.wiki_username, vector, processor=actor.username
        )
        _comparison(job.wiki_username, embed_tags=True)
        _log(
            "lookup",
            lookup_name=job.wiki_username,
            useful_count=job.useful_count,
            actor=actor,
        )
        job.status = "done"
        job.finished_at = datetime.now(timezone.utc)
        db.session.commit()
        return jsonify({"status": "done"})

    @app.post("/tags")
    def add_tag():
        subject = normalize_username(request.form.get("subject", ""))
        tag_name = request.form.get("tag", "").strip()
        selected = [
            normalize_username(name)
            for name in request.form.getlist("account")
            if name.strip()
        ]
        if tag_name and "/" not in tag_name and "," not in tag_name and selected:
            tag = _tag(tag_name, cosmetic=request.form.get("cosmetic", "") == "on")
            actor = _current_actor()
            added: list[str] = []
            for name in selected:
                exists = TagMember.query.filter_by(
                    tag_id=tag.id, wiki_username=name
                ).one_or_none()
                if exists is not None:
                    continue
                db.session.add(
                    TagMember(
                        tag_id=tag.id,
                        wiki_username=name,
                        added_by_id=actor.id,
                        added_at=datetime.now(timezone.utc),
                    )
                )
                added.append(name)
            if added:
                _log(
                    "tag_add",
                    tag=tag,
                    users=added,
                    added_by={name: actor.username for name in added},
                )
                _refresh_tag_style(tag)
            db.session.commit()
        return redirect(url_for("index", user=subject) if subject else url_for("index"))

    @app.get("/tag/<name>")
    def tag_page(name: str):
        tag = Tag.query.filter_by(name=name).one_or_none()
        if tag is None:
            abort(404)
        members = (
            TagMember.query.filter_by(tag_id=tag.id)
            .order_by(TagMember.wiki_username)
            .all()
        )
        names = [member.wiki_username for member in members]
        rows = _person_rows(names, similarities=None)
        scores = _outlier_scores(names)
        for row in rows:
            row.similarity = scores.get(row.name)
        rows.sort(
            key=lambda row: row.similarity if row.similarity is not None else -1,
            reverse=True,
        )
        return render_template(
            "tag.html", tag=tag, rows=rows, member_count=len(members)
        )

    @app.post("/tag/<name>/notes")
    def save_notes(name: str):
        tag = Tag.query.filter_by(name=name).one_or_none()
        if tag is None:
            abort(404)
        updated = request.form.get("notes", "").replace("\r\n", "\n")
        previous = tag.notes or ""
        if updated != previous:
            tag.notes = updated
            _log("note_edit", tag=tag, before=previous, after=updated)
            db.session.commit()
        return redirect(url_for("tag_page", name=tag.name))

    @app.post("/tag/<name>/cosmetic")
    def set_cosmetic(name: str):
        tag = Tag.query.filter_by(name=name).one_or_none()
        if tag is None:
            abort(404)
        cosmetic = request.form.get("cosmetic", "") == "on"
        if bool(tag.cosmetic) != cosmetic:
            tag.cosmetic = cosmetic
            if cosmetic:
                _forget_tag_scores(tag.name)
            else:
                _enqueue_tag_job(tag.name)
            db.session.commit()
        return redirect(url_for("tag_page", name=tag.name))

    @app.post("/tag/<name>/delete")
    def delete_tag(name: str):
        tag = Tag.query.filter_by(name=name).one_or_none()
        if tag is None:
            abort(404)
        if TagMember.query.filter_by(tag_id=tag.id).count():
            abort(400)
        label = tag.name
        _log("tag_delete", tag=tag)
        db.session.flush()
        for entry in LogEntry.query.filter_by(tag_id=tag.id):
            data = json.loads(entry.payload or "{}")
            if not isinstance(data, dict):
                data = {}
            data["tag_name"] = label
            entry.payload = json.dumps(data)
            entry.tag_id = None
        db.session.delete(tag)
        db.session.commit()
        return redirect(url_for("tags"))

    @app.get("/tags")
    def tags():
        found = Tag.query.order_by(Tag.name).all()
        listed = [
            (tag, TagMember.query.filter_by(tag_id=tag.id).count()) for tag in found
        ]
        return render_template("tags.html", tags=listed)

    @app.post("/tag/<name>/remove")
    def remove_tag(name: str):
        tag = Tag.query.filter_by(name=name).one_or_none()
        if tag is None:
            abort(404)
        selected = {
            normalize_username(item)
            for item in request.form.getlist("account")
            if item.strip()
        }
        members = [
            member
            for member in TagMember.query.filter_by(tag_id=tag.id)
            if member.wiki_username in selected
        ]
        if members:
            added_by = {
                member.wiki_username: member.added_by.username for member in members
            }
            users = [member.wiki_username for member in members]
            for member in members:
                db.session.delete(member)
            _log("tag_remove", tag=tag, users=users, added_by=added_by)
            _refresh_tag_style(tag)
            db.session.commit()
        return redirect(url_for("tag_page", name=tag.name))

    @app.get("/log")
    def log():
        page_num = request.args.get("page", 1, type=int)
        page = LogEntry.query.order_by(LogEntry.id.desc()).paginate(
            page=page_num, per_page=50, error_out=False
        )
        return render_template("log.html", page=page)

    @app.post("/log/<int:entry_id>/undo")
    def undo(entry_id: int):
        entry = db.session.get(LogEntry, entry_id)
        if (
            entry is None
            or entry.undone
            or entry.action not in {"tag_add", "tag_remove", "undo_add", "undo_remove"}
        ):
            abort(404)
        tag = entry.tag
        if tag is None:
            abort(404)
        actor = _current_actor()
        users = entry.names()
        if entry.action in {"tag_add", "undo_remove"}:
            for member in TagMember.query.filter_by(tag_id=tag.id):
                if member.wiki_username in users:
                    db.session.delete(member)
            _log("undo_add", tag=tag, users=users, added_by=entry.added_by_map())
        else:
            restored: list[str] = []
            adders = entry.added_by_map()
            for name in users:
                if TagMember.query.filter_by(
                    tag_id=tag.id, wiki_username=name
                ).one_or_none():
                    continue
                adder = _actor(adders.get(name) or actor.username)
                db.session.add(
                    TagMember(
                        tag_id=tag.id,
                        wiki_username=name,
                        added_by_id=adder.id,
                        added_at=datetime.now(timezone.utc),
                    )
                )
                restored.append(name)
            _log("undo_remove", tag=tag, users=restored or users, added_by=adders)
        _refresh_tag_style(tag)
        entry.undone = True
        db.session.commit()
        return redirect(url_for("log"))

    @app.get("/log/<int:entry_id>/diff")
    def log_diff(entry_id: int):
        entry = db.session.get(LogEntry, entry_id)
        if entry is None or entry.action != "note_edit":
            abort(404)
        return render_template("diff.html", entry=entry)

    return app


def csrf_token() -> str:
    token = session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf"] = token
    return token


def _services() -> dict:
    gone = current_app.extensions["gone"]
    if gone.get("client") is None:
        print("loading wiki client and encoder", flush=True)
        if os.environ.get("GONEFISHING_REMOTE_ENCODER") == "1":
            from encoder_service import ensure_encoder_server

            ensure_encoder_server()
        _load_services(current_app, gone["env"], gone["root"])
        print("wiki client and encoder ready", flush=True)
    return current_app.extensions["gone"]


def _load_services(app: Flask, env: dict[str, str], root: Path) -> None:
    config = PipelineConfig(
        user_agent="gonefishing/0.1 by en:User:MSK <thewonderfulworldofpotatoes at gmail dot com>",
        content=GroupThresholds(
            min_sizediff=80, min_prose_chars=80, min_sentences=1, min_prose_ratio=0.0
        ),
        is_bot=True,
    )
    data = app.extensions["gone"]["data"]
    cache = RevisionCache(data / "revisions.sqlite")
    client = WikiClient(config, cache)
    client.login(env["USER"], env["PASS"])
    namespaces = included_namespaces(client)
    store = AuthorStore(data / "authors.sqlite")
    corpus = data / "good_diffs"
    if not corpus.is_dir():
        corpus = root / "data" / "good_diffs"
    store.import_corpus(corpus)
    if os.environ.get("GONEFISHING_REMOTE_ENCODER") == "1":
        from encoder_service import RemoteLuarEncoder

        encoder = RemoteLuarEncoder()
    else:
        encoder = LuarEncoder(
            luar_onnx_path(root),
            root / "data" / "luar_mud_lora" / "tokenizer.json",
        )
    app.extensions["gone"].update(
        {
            "config": config,
            "client": client,
            "namespaces": namespaces,
            "store": store,
            "encoder": encoder,
        }
    )


def _privileged(username: str) -> bool:
    gone = current_app.extensions["gone"]
    if named_allowed(username, gone["allowlist"]):
        return True
    client = _services().get("client")
    if client is None:
        return False
    try:
        if "checkuser" in client.local_groups(username):
            return True
        if "steward" in client.global_groups(username):
            return True
    except Exception:
        return False
    return False


def _actor(username: str) -> AppUser:
    found = AppUser.query.filter_by(username=username).one_or_none()
    if found is not None:
        return found
    found = AppUser(username=username)
    db.session.add(found)
    db.session.commit()
    return found


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _local_job_is_stale(job: LookupJob) -> bool:
    if job.created_at is None:
        return False
    return _aware(job.created_at) < datetime.now(timezone.utc) - LOCAL_JOB_TTL


def _expire_local_jobs() -> None:
    now = datetime.now(timezone.utc)
    changed = False
    jobs = LookupJob.query.filter(
        LookupJob.local.is_(True),
        LookupJob.status.in_(("queued", "running", "ready")),
    ).all()
    for job in jobs:
        if not _local_job_is_stale(job):
            continue
        job.status = "not done"
        job.finished_at = now
        changed = True
    if changed:
        db.session.commit()


def _supplied_csrf() -> str:
    return request.form.get("csrf") or request.headers.get("X-CSRF-Token") or ""


def _local_waiting(username: str) -> bool:
    return (
        LookupJob.query.filter(
            LookupJob.wiki_username == username,
            LookupJob.local.is_(True),
            LookupJob.status == "ready",
        ).first()
        is not None
    )


def _owned_local_job(qid: str) -> LookupJob:
    _expire_local_jobs()
    job = LookupJob.query.filter_by(public_id=qid, local=True).one_or_none()
    if job is None or not qid:
        abort(404)
    if job.actor_id != _current_actor().id:
        abort(403)
    return job


def _local_vector(payload: object) -> np.ndarray | None:
    if not isinstance(payload, list) or len(payload) != VECTOR_DIM:
        return None
    try:
        vector = np.asarray(payload, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (VECTOR_DIM,) or not np.isfinite(vector).all():
        return None
    return vector


def _current_actor() -> AppUser:
    actor = db.session.get(AppUser, session.get("user_id"))
    if actor is None:
        abort(403)
    return actor


def _tag(name: str, *, cosmetic: bool = False) -> Tag:
    cleaned = name.strip()
    existing = Tag.query.filter(
        db.func.lower(Tag.name) == cleaned.casefold()
    ).one_or_none()
    if existing is not None:
        return existing
    created = Tag(name=cleaned, cosmetic=cosmetic)
    db.session.add(created)
    db.session.flush()
    return created


def _refresh_tag_style(tag: Tag) -> None:
    if tag.cosmetic:
        return
    _forget_tag_scores(tag.name)
    _enqueue_tag_job(tag.name)


def _log(
    action: str,
    *,
    tag: Tag | None = None,
    users: list[str] | None = None,
    added_by: dict[str, str] | None = None,
    lookup_name: str | None = None,
    useful_count: int | None = None,
    before: str | None = None,
    after: str | None = None,
    actor: AppUser | None = None,
) -> None:
    payload = {"users": users or [], "added_by": added_by or {}}
    if tag is not None:
        payload["tag_name"] = tag.name
    if before is not None:
        payload["before"] = before
    if after is not None:
        payload["after"] = after
    entry = LogEntry(
        actor_id=(actor or _current_actor()).id,
        action=action,
        tag_id=None if tag is None else tag.id,
        lookup_name=lookup_name,
        useful_count=useful_count,
        payload=json.dumps(payload),
        undone=False,
        created_at=datetime.now(timezone.utc),
    )
    db.session.add(entry)


def _start_lookup_worker(app: Flask) -> None:
    def loop() -> None:
        with app.app_context():
            while True:
                _expire_local_jobs()
                job = (
                    LookupJob.query.filter_by(status="queued")
                    .order_by(LookupJob.id)
                    .first()
                )
                if job is None:
                    if _backfill_one_signal():
                        db.session.remove()
                        continue
                    db.session.remove()
                    time.sleep(1)
                    continue
                job_id = job.id
                kind = job.kind or "lookup"
                claimed = LookupJob.query.filter_by(id=job_id, status="queued").update(
                    {
                        "status": "running",
                        "started_at": datetime.now(timezone.utc),
                    }
                )
                db.session.commit()
                if claimed != 1:
                    continue
                try:
                    if kind == "tag":
                        _execute_tag(job_id)
                    else:
                        _execute_lookup(job_id)
                except Exception:
                    db.session.rollback()
                    failed = db.session.get(LookupJob, job_id)
                    if failed is not None and failed.status == "running":
                        failed.status = "error"
                        failed.error = traceback.format_exc()[-2000:]
                        failed.finished_at = datetime.now(timezone.utc)
                        db.session.commit()
                finally:
                    db.session.remove()

    with app.app_context():
        LookupJob.query.filter_by(status="running").update({"status": "queued"})
        db.session.commit()
    threading.Thread(target=loop, name="lookup-jobs", daemon=True).start()


def _execute_lookup(job_id: int) -> None:
    job = db.session.get(LookupJob, job_id)
    if job is None:
        return
    username = job.wiki_username
    refresh = job.refresh
    local = bool(job.local)
    actor = db.session.get(AppUser, job.actor_id)
    store = _services()["store"]
    if refresh:
        WikiProfile.query.filter_by(wiki_username=username).delete()
        db.session.commit()
    if refresh or not store.has_user(username):
        kept = _collect(username)
    else:
        kept = store.count(username)
    _ensure_signals(username)
    if local:
        finished = db.session.get(LookupJob, job_id)
        if finished is None:
            return
        if kept <= 0:
            finished.status = "error"
            finished.error = "No usable edits were stored for that account."
            finished.finished_at = datetime.now(timezone.utc)
        elif _local_job_is_stale(finished):
            finished.status = "not done"
            finished.finished_at = datetime.now(timezone.utc)
        else:
            finished.status = "ready"
            finished.useful_count = kept
        db.session.commit()
        return
    _comparison(username, embed_tags=True)
    _log("lookup", lookup_name=username, useful_count=kept, actor=actor)
    finished = db.session.get(LookupJob, job_id)
    if finished is None:
        db.session.commit()
        return
    finished.status = "done"
    finished.useful_count = kept
    finished.finished_at = datetime.now(timezone.utc)
    db.session.commit()


def _busy_notice(running: LookupJob, subject: str) -> str:
    if running.kind == "tag":
        busy = f"An update of the tag {running.tag_name}"
    else:
        busy = f"A comparison of {running.wiki_username}"
    return f"{busy} is already running. {subject} is waiting in the queue."


def _enqueue_tag_job(tag_name: str, actor_id: int | None = None) -> None:
    queued = LookupJob.query.filter_by(
        kind="tag", tag_name=tag_name, status="queued"
    ).first()
    if queued is not None:
        return
    db.session.add(
        LookupJob(
            actor_id=actor_id if actor_id is not None else _current_actor().id,
            wiki_username="",
            kind="tag",
            tag_name=tag_name,
            refresh=False,
            status="queued",
            created_at=datetime.now(timezone.utc),
        )
    )


def _execute_tag(job_id: int) -> None:
    job = db.session.get(LookupJob, job_id)
    if job is None:
        return
    _rebuild_tag(job.tag_name)
    finished = db.session.get(LookupJob, job_id)
    if finished is None:
        db.session.commit()
        return
    finished.status = "done"
    finished.finished_at = datetime.now(timezone.utc)
    db.session.commit()


def _rebuild_tag(tag_name: str) -> None:
    gone = _services()
    store: AuthorStore = gone["store"]
    tag = Tag.query.filter_by(name=tag_name).one_or_none()
    if tag is None or tag.cosmetic:
        store.drop_tag_scores(tag_name)
        return
    members = [
        member.wiki_username for member in TagMember.query.filter_by(tag_id=tag.id)
    ]
    members_key = _member_key(members)
    parts: dict[str, tuple[np.ndarray, int]] = {}
    for member in members:
        prose = store.prose(member)
        if not prose:
            continue
        vectors = gone["encoder"].embed_many([[text] for text in prose])
        parts[member] = (np.asarray(vectors, dtype=np.float64).sum(axis=0), len(prose))
    current = [
        member.wiki_username for member in TagMember.query.filter_by(tag_id=tag.id)
    ]
    if _member_key(current) != members_key:
        return
    store.put_tag_parts(tag_name, members_key, parts)


def _ensure_tag_notes() -> None:
    rows = db.session.execute(db.text("PRAGMA table_info(tags)")).fetchall()
    names = {row[1] for row in rows}
    if rows and "notes" not in names:
        db.session.execute(
            db.text("ALTER TABLE tags ADD COLUMN notes TEXT NOT NULL DEFAULT ''")
        )
    if rows and "cosmetic" not in names:
        db.session.execute(
            db.text("ALTER TABLE tags ADD COLUMN cosmetic INTEGER NOT NULL DEFAULT 0")
        )
    if rows:
        db.session.commit()


def _ensure_lookup_jobs() -> None:
    rows = db.session.execute(db.text("PRAGMA table_info(lookup_jobs)")).fetchall()
    names = {row[1] for row in rows}
    if not rows:
        return
    if "kind" not in names:
        db.session.execute(
            db.text(
                "ALTER TABLE lookup_jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'lookup'"
            )
        )
    if "tag_name" not in names:
        db.session.execute(
            db.text(
                "ALTER TABLE lookup_jobs ADD COLUMN tag_name TEXT NOT NULL DEFAULT ''"
            )
        )
    if "local" not in names:
        db.session.execute(
            db.text(
                "ALTER TABLE lookup_jobs ADD COLUMN local INTEGER NOT NULL DEFAULT 0"
            )
        )
    if "public_id" not in names:
        db.session.execute(
            db.text(
                "ALTER TABLE lookup_jobs ADD COLUMN public_id TEXT NOT NULL DEFAULT ''"
            )
        )
    db.session.commit()


def _forget_tag_scores(tag_name: str) -> None:
    store = _services().get("store")
    if store is not None:
        store.drop_tag_scores(tag_name)


_wiki_lock = threading.Lock()


def _collect(username: str) -> int:
    gone = _services()
    with _wiki_lock:
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "user.jsonl"
            scanned: list = []
            kept = collect_user(
                gone["client"],
                username,
                gone["namespaces"],
                gone["config"],
                dest,
                max_useful=MAX_USEFUL,
                max_scanned=MAX_USEFUL * 10,
                facts=scanned,
            )
            loaded = [
                Document(username, document.revid, document.pageid, document.prose)
                for docs in load_documents(dest).values()
                for document in docs
            ]
        if loaded:
            gone["store"].replace_user(loaded)
        gone["store"].replace_facts(
            username,
            [fact_from_contrib(contrib, gone["config"]) for contrib in scanned],
        )
        return kept


def _ensure_vector(user: str):
    gone = _services()
    store: AuthorStore = gone["store"]
    cached = store.vector(user)
    if cached is not None:
        return cached
    prose = store.prose(user)[:MAX_USEFUL]
    if not prose:
        return None
    store.put_vector(user, gone["encoder"].embed_episode(prose))
    return store.vector(user)


def _memberships(names: list[str]) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    if not names:
        return grouped
    members = TagMember.query.filter(TagMember.wiki_username.in_(names)).all()
    for member in members:
        grouped.setdefault(member.wiki_username, []).append(
            (member.tag.name, member.added_by.username)
        )
    for items in grouped.values():
        items.sort(key=lambda item: item[0].casefold())
    return grouped


def _facts(names: list[str]) -> dict[str, dict]:
    if not names:
        return {}
    rows = WikiProfile.query.filter(WikiProfile.wiki_username.in_(names)).all()
    indexed = {_profile_key(row): _profile_info(row) for row in rows}
    missing = [name for name in names if name_key(name) not in indexed]
    client = _services().get("client")
    if missing and client is not None:
        try:
            raw = client.user_facts(missing)
        except Exception:
            raw = {}
        for info in raw.values():
            name = str(info.get("name") or "")
            if not name:
                continue
            profile = WikiProfile(
                wiki_username=name,
                editcount=int(info.get("editcount") or 0),
                registration=str(info.get("registration") or ""),
                blockedtimestamp=str(info.get("blockedtimestamp") or ""),
                blockedby=str(info.get("blockedby") or ""),
                blockreason=str(info.get("blockreason") or ""),
            )
            db.session.merge(profile)
            indexed[name_key(name)] = _profile_info(profile)
        db.session.commit()
    return indexed


def _profile_key(row: WikiProfile) -> str:
    return name_key(row.wiki_username)


def _profile_info(row: WikiProfile) -> dict:
    return {
        "editcount": row.editcount,
        "registration": row.registration,
        "blockedtimestamp": row.blockedtimestamp,
        "blockedby": row.blockedby,
        "blockreason": row.blockreason,
    }


def _outlier_scores(names: list[str]) -> dict[str, float]:
    store: AuthorStore | None = current_app.extensions["gone"].get("store")
    if store is None:
        return {}
    vectors: dict[str, np.ndarray] = {}
    for name in names:
        vector = store.vector(name)
        if vector is not None:
            vectors[name] = vector
    return centroid_distances(vectors)


def _person_rows(
    names: list[str], similarities: dict[str, float] | None
) -> list[ViewRow]:
    now = datetime.now(timezone.utc)
    store: AuthorStore | None = _services().get("store")
    facts = _facts(names)
    tags = _memberships(names)
    counts = {} if store is None else store.counts()
    local = {} if store is None else store.local_processors(names)
    rows: list[ViewRow] = []
    for name in names:
        info = facts.get(name_key(name), {})
        useful = counts.get(name, 0)
        links = account_links(name)
        score = None if similarities is None else similarities.get(name)
        registered = parse_timestamp(str(info.get("registration") or ""))
        rows.append(
            ViewRow(
                kind="person",
                name=name,
                similarity=score,
                local_by="" if store is None else local.get(name, ""),
                stats=account_stats(
                    int(info.get("editcount") or 0),
                    str(info.get("registration") or ""),
                    useful,
                    now,
                ),
                block=block_text(info),
                edits=int(info.get("editcount") or 0),
                created=registered.date().isoformat() if registered else "",
                blocked=bool(info.get("blockedtimestamp")),
                tags=tags.get(name, []),
                talk=links["talk"],
                contribs=links["contribs"],
                centralauth=links["centralauth"],
                blocklog=links["blocklog"],
            )
        )
    return rows


def _member_key(names: list[str]) -> str:
    return "\n" + "\n".join(sorted(names, key=str.casefold)) + "\n"


def _comparison(subject: str, embed_tags: bool = False) -> list[ViewRow]:
    gone = _services()
    store: AuthorStore = gone["store"]
    query_vector = _ensure_vector(subject)
    if query_vector is None:
        return []
    authors = [
        author for author in store.users() if name_key(author) != name_key(subject)
    ]
    scores = store.scores_for(subject)
    missing = [author for author in authors if author not in scores]
    busy = {
        job.wiki_username
        for job in LookupJob.query.filter(
            LookupJob.local.is_(True),
            LookupJob.status.in_(("queued", "running", "ready")),
        ).all()
    }
    fresh: dict[str, float] = {}
    for author in missing:
        if author in busy:
            continue
        vector = _ensure_vector(author)
        if vector is None:
            continue
        fresh[author] = float(np.dot(query_vector, vector))
    if fresh:
        store.put_scores(subject, fresh)
        scores.update(fresh)
    rows = _person_rows([author for author in authors if author in scores], scores)
    for row in rows:
        _apply_blend(row, store, subject)
    subject_key = name_key(subject)
    for tag in Tag.query.order_by(Tag.name).all():
        if tag.cosmetic:
            continue
        members = [
            member.wiki_username for member in TagMember.query.filter_by(tag_id=tag.id)
        ]
        if not members:
            continue
        members_key = _member_key(members)
        excluded = [member for member in members if name_key(member) != subject_key]
        cached = (
            store.tag_score(subject, tag.name, _member_key(excluded))
            if excluded
            else None
        )
        parts = store.tag_parts(tag.name, members_key)
        if parts is None and cached is None and embed_tags:
            _rebuild_tag(tag.name)
            parts = store.tag_parts(tag.name, members_key)
        if parts is not None:
            sums = []
            counts = []
            for member, (total, count) in parts.items():
                if name_key(member) == subject_key:
                    continue
                sums.append(total)
                counts.append(count)
            score = tag_similarity(query_vector, sums, counts)
            if score is None:
                continue
            rows.append(ViewRow(kind="tag", name=tag.name, similarity=score))
            continue
        if cached is None:
            if not embed_tags and excluded:
                rows.append(ViewRow(kind="tag", name=tag.name, similarity=None))
            continue
        rows.append(ViewRow(kind="tag", name=tag.name, similarity=cached))
    rows.sort(
        key=lambda row: row.similarity if row.similarity is not None else -1,
        reverse=True,
    )
    subject_row = _person_rows([subject], {subject: 1.0})[0]
    subject_row.pinned = True
    _apply_blend(subject_row, store, subject)
    rows.insert(0, subject_row)
    return rows


def _backfill_facts(username: str) -> None:
    gone = _services()
    with _wiki_lock:
        contribs = gone["client"].recent_contribs(username, gone["namespaces"], limit=2500)
    gone["store"].replace_facts(
        username,
        [fact_from_contrib(contrib, gone["config"]) for contrib in contribs],
    )


def _backfill_one_signal() -> bool:
    store: AuthorStore | None = current_app.extensions["gone"].get("store")
    if store is None:
        return False
    missing = store.users_missing_signals(need_topics=True)
    if not missing:
        return False
    username = missing[0]
    print(f"backfilling signals for {username}", flush=True)
    _ensure_signals(username)
    return True


def _ensure_signals(username: str) -> None:
    store: AuthorStore = _services()["store"]
    if not store.edit_facts(username):
        _backfill_facts(username)
    facts = store.edit_facts(username)
    if not facts:
        if store.hour_histogram(username) is None:
            store.put_hour_histogram(username, np.zeros(24))
        if store.topic_vector(username) is None:
            store.put_topic_vector(username, {})
        return
    if store.hour_histogram(username) is None:
        bins = hour_histogram(facts)
        store.put_hour_histogram(username, bins if bins is not None else np.zeros(24))
    if store.topic_vector(username) is None:
        topics = collect_topic_vector(facts)
        if topics is not None:
            store.put_topic_vector(username, topics)


def _apply_blend(row: ViewRow, store: AuthorStore, subject: str) -> None:
    if row.kind != "person" or row.similarity is None:
        return
    if name_key(row.name) == name_key(subject):
        topic = 1.0 if store.topic_vector(subject) else None
        time_score = (
            1.0
            if (hours := store.hour_histogram(subject)) is not None and float(hours.sum()) > 0
            else None
        )
    else:
        topic = cosine_dicts(store.topic_vector(subject), store.topic_vector(row.name))
        time_score = cosine_histograms(
            store.hour_histogram(subject), store.hour_histogram(row.name)
        )
    score, detail = blend_similarity(row.similarity, topic, time_score)
    row.similarity = score
    row.similarity_detail = detail
