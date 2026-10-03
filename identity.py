"""Username normalization and the static allowlist."""

from datetime import datetime, timezone
from pathlib import Path


def normalize_username(name: str) -> str:
    return name.replace("_", " ").strip()


def name_key(name: str) -> str:
    return normalize_username(name).casefold()


def clerk_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        key = name_key(line)
        if key:
            keys.add(key)
    return keys


def named_allowed(username: str, allowlist: set[str]) -> bool:
    key = name_key(username)
    return key in allowlist


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_day(moment: datetime) -> str:
    return f"{moment.day} {moment.strftime('%B')} {moment.year}"


def age_phrase(registered: datetime, now: datetime) -> str:
    days = (now.date() - registered.date()).days
    if days >= 365:
        years = days // 365
        unit = "year" if years == 1 else "years"
        return f"{years} {unit}"
    if days >= 30:
        months = days // 30
        unit = "month" if months == 1 else "months"
        return f"{months} {unit}"
    unit = "day" if days == 1 else "days"
    return f"{days} {unit}"


def account_stats(editcount: int, registration: str, useful: int, now: datetime) -> str:
    useful_bit = f"{useful} useful edits"
    registered = parse_timestamp(registration)
    if registered is None:
        return f"{editcount} ({useful_bit})"
    return f"<span title='{useful_bit}' style='text-decoration: underline dotted;'>{editcount}</span> since <span title='{age_phrase(registered, now)}' style='text-decoration: underline dotted;'>{format_day(registered)}</span>"


def block_text(info: dict) -> str:
    stamp = str(info.get("blockedtimestamp") or "")
    if not stamp:
        return ""
    when = parse_timestamp(stamp)
    admin = str(info.get("blockedby") or "")
    reason = str(info.get("blockreason") or "")
    return f"Blocked {format_day(when)} {" by " + admin if admin else ""}{" with reason \"" + reason + "\"" if reason else ""}"
