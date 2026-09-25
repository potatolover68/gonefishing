"""Reject edits XTools would count as semi-automated on English Wikipedia.

Rules come from MediaWiki:XTools-AutoEdits.json. Global, ``en``, and
``en.wikipedia.org`` entries are merged the way XTools does: a wiki regex is
added to the global one, and other fields on the more specific entry replace
the broader ones. Tools marked ``contribs`` are listed by XTools but not
counted as automated, so they are skipped here too.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "automated.json"
_LAYERS = ("global", "en", "en.wikipedia.org")


@dataclass(frozen=True)
class AutoEditRule:
    name: str
    regex: re.Pattern[str] | None
    tags: frozenset[str]
    tag_excludes: frozenset[str]
    namespaces: frozenset[int] | None
    talk_namespaces: bool


def _as_tags(values: dict) -> list[str]:
    tags = values.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    single = values.get("tag")
    if isinstance(single, str):
        tags = [*tags, single]
    return [tag for tag in tags if tag]


def _merge_tool(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "regex" and merged.get("regex"):
            merged["regex"] = f"(?:{merged['regex']})|(?:{value})"
        else:
            merged[key] = value
    return merged


def _namespace_ok(ns: int, rule: AutoEditRule) -> bool:
    if not rule.namespaces:
        return True
    if ns in rule.namespaces:
        return True
    return rule.talk_namespaces and ns % 2 == 1


def matches_autoedit(ns: int, comment: str, tags: frozenset[str], rules: tuple[AutoEditRule, ...]) -> str | None:
    text = comment or ""
    for rule in rules:
        if not _namespace_ok(ns, rule):
            continue
        if rule.regex is not None and rule.regex.search(text):
            return rule.name
        if rule.tags and (tags & rule.tags) and not (tags & rule.tag_excludes):
            return rule.name
    return None


def load_rules(path: Path | None = None) -> tuple[AutoEditRule, ...]:
    source = Path(path) if path is not None else DEFAULT_PATH
    return _load_cached(str(source.resolve()))


@lru_cache(maxsize=4)
def _load_cached(path: str) -> tuple[AutoEditRule, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tools: dict[str, dict] = {}
    for layer in _LAYERS:
        for name, values in payload.get(layer, {}).items():
            if not isinstance(values, dict):
                continue
            tools[name] = _merge_tool(tools.get(name, {}), values)

    rules: list[AutoEditRule] = []
    for name, values in tools.items():
        if values.get("contribs"):
            continue
        tags = frozenset(_as_tags(values))
        pattern = values.get("regex")
        compiled = re.compile(pattern, re.IGNORECASE) if pattern else None
        revert = values.get("revert")
        if isinstance(revert, str) and revert:
            extra = re.compile(revert, re.IGNORECASE)
            compiled = extra if compiled is None else re.compile(
                f"(?:{compiled.pattern})|(?:{extra.pattern})",
                re.IGNORECASE,
            )
        namespaces = values.get("namespaces")
        if not tags and compiled is None:
            continue
        rules.append(
            AutoEditRule(
                name=name,
                regex=compiled,
                tags=tags,
                tag_excludes=frozenset(values.get("tag_excludes") or []),
                namespaces=frozenset(int(item) for item in namespaces) if namespaces else None,
                talk_namespaces=bool(values.get("talk_namespaces")),
            )
        )
    return tuple(rules)
