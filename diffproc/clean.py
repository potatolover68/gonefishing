import re

import mwparserfromhell

from diffproc.config import PipelineConfig

_TIMESTAMP = re.compile(r"\d{2}:\d{2}, \d{1,2} \w+ \d{4} \(UTC\)")
_USER_LINK = re.compile(r"\[\[User(?: talk)?:", re.IGNORECASE)
_PRECEDING_USER_LINK = re.compile(r"\[\[User(?: talk)?:.*?\]\]$", re.IGNORECASE)
_SIGNATURE_CHROME = re.compile(r"</?(?:span|font|small)\b[^>]*>", re.IGNORECASE)
_INDENT = re.compile(r"^[:*#]+")
_WS = re.compile(r"\s+")
_NAMESPACE_PREFIXES = ("file:", "image:", "category:")


def _strip_indent(line: str) -> str:
    return _INDENT.sub("", line, count=1).lstrip()


def _strip_signature_line(line: str) -> str:
    timestamp = _TIMESTAMP.search(line)
    if timestamp is None:
        line = _SIGNATURE_CHROME.sub("", line)
        return _strip_indent(line)
    start, end = timestamp.span()
    links = [
        match.start() for match in _USER_LINK.finditer(line) if match.start() < start
    ]
    if not links:
        line = line[:start] + line[end:]
    else:
        # The last user link is often [[User talk:]]. Also drop the user link
        # immediately before it so a standard signature leaves no display name.
        head = line[: links[-1]].rstrip(" \t").rstrip("(").rstrip(" \t")
        preceding = _PRECEDING_USER_LINK.search(head)
        if preceding is not None:
            head = head[: preceding.start()]
        line = head + line[end:]
    line = _SIGNATURE_CHROME.sub("", line)
    return _strip_indent(line)


def strip_signatures(text: str) -> str:
    return "\n".join(_strip_signature_line(line) for line in text.split("\n"))


def _remove_all(code: mwparserfromhell.wikicode.Wikicode, nodes) -> None:
    for node in nodes:
        try:
            code.remove(node)
        except ValueError:
            continue


def _namespaced_link(link, prefixes: frozenset[str]) -> bool:
    title = str(link.title).strip().lstrip(":")
    lowered = title.casefold()
    if lowered.startswith(_NAMESPACE_PREFIXES):
        return True
    if ":" not in title:
        return False
    prefix = title.split(":", 1)[0].strip().casefold()
    return prefix in prefixes


def clean_paragraph(raw: str, ns_group: str, config: PipelineConfig) -> str | None:
    text = strip_signatures(raw) if ns_group == "discussion" else raw
    if "{{" not in text and "<" not in text and "[[" not in text:
        prose = _WS.sub(" ", text).strip()
        return prose or None
    code = mwparserfromhell.parse(text)
    _remove_all(code, list(code.filter_templates(recursive=True)))
    removed = {name.casefold() for name in config.removed_tags}
    tags = [
        tag
        for tag in code.filter_tags(recursive=True)
        if str(tag.tag).strip().casefold() in removed
    ]
    _remove_all(code, tags)
    _remove_all(code, list(code.filter_comments(recursive=True)))
    _remove_all(code, list(code.filter_headings(recursive=True)))
    links = [
        link
        for link in code.filter_wikilinks(recursive=True)
        if _namespaced_link(
            link, {prefix.casefold() for prefix in config.interwiki_prefixes}
        )
    ]
    _remove_all(code, links)
    prose = _WS.sub(" ", code.strip_code()).strip()
    return prose or None
