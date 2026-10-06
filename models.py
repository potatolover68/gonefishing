from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class AppUser(db.Model):
    __tablename__ = "app_users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)


class Tag(db.Model):
    __tablename__ = "tags"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    notes = db.Column(db.Text, nullable=False, default="")


class TagMember(db.Model):
    __tablename__ = "tag_members"
    __table_args__ = (db.UniqueConstraint("tag_id", "wiki_username"),)

    id = db.Column(db.Integer, primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey("tags.id"), nullable=False)
    wiki_username = db.Column(db.String(255), nullable=False)
    added_by_id = db.Column(db.Integer, db.ForeignKey("app_users.id"), nullable=False)
    added_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    tag = db.relationship("Tag")
    added_by = db.relationship("AppUser")


class LookupJob(db.Model):
    __tablename__ = "lookup_jobs"

    id = db.Column(db.Integer, primary_key=True)
    actor_id = db.Column(db.Integer, db.ForeignKey("app_users.id"), nullable=False)
    wiki_username = db.Column(db.String(255), nullable=False)
    kind = db.Column(db.String(16), nullable=False, default="lookup")
    tag_name = db.Column(db.String(255), nullable=False, default="")
    refresh = db.Column(db.Boolean, nullable=False, default=False)
    local = db.Column(db.Boolean, nullable=False, default=False)
    public_id = db.Column(db.String(64), nullable=False, default="")
    status = db.Column(db.String(16), nullable=False, default="queued")
    useful_count = db.Column(db.Integer, nullable=True)
    error = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    actor = db.relationship("AppUser")

    def stamp(self) -> str:
        moment = self.created_at
        if moment is None:
            return ""
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return f"{moment.day} {moment.strftime('%B %Y, %H:%M')} UTC"


class WikiProfile(db.Model):
    __tablename__ = "wiki_profiles"

    wiki_username = db.Column(db.String(255), primary_key=True)
    editcount = db.Column(db.Integer, nullable=False, default=0)
    registration = db.Column(db.String(64), nullable=False, default="")
    blockedtimestamp = db.Column(db.String(64), nullable=False, default="")
    blockedby = db.Column(db.String(255), nullable=False, default="")
    blockreason = db.Column(db.Text, nullable=False, default="")


class LogEntry(db.Model):
    __tablename__ = "log_entries"

    id = db.Column(db.Integer, primary_key=True)
    actor_id = db.Column(db.Integer, db.ForeignKey("app_users.id"), nullable=False)
    action = db.Column(db.String(32), nullable=False)
    tag_id = db.Column(db.Integer, db.ForeignKey("tags.id"), nullable=True)
    lookup_name = db.Column(db.String(255), nullable=True)
    useful_count = db.Column(db.Integer, nullable=True)
    payload = db.Column(db.Text, nullable=False, default="{}")
    undone = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    actor = db.relationship("AppUser")
    tag = db.relationship("Tag")

    def names(self) -> list[str]:
        return [str(name) for name in self._payload().get("users") or []]

    def added_by_map(self) -> dict[str, str]:
        raw = self._payload().get("added_by") or {}
        return {str(name): str(adder) for name, adder in raw.items()}

    def tag_label(self) -> str:
        if self.tag is not None:
            return self.tag.name
        return str(self._payload().get("tag_name") or "")

    def note_before(self) -> str:
        return str(self._payload().get("before") or "")

    def note_after(self) -> str:
        return str(self._payload().get("after") or "")

    def stamp(self) -> str:
        moment = self.created_at
        if moment is None:
            return ""
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return f"{moment.day} {moment.strftime('%B %Y, %H:%M')} UTC"

    def _payload(self) -> dict:
        import json

        data = json.loads(self.payload or "{}")
        return data if isinstance(data, dict) else {}
