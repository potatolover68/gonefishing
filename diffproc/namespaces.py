from collections.abc import Mapping

CONTENT_NAMES = frozenset({"", "Wikipedia", "Draft"})


def ns_group(ns: int, content_ids: frozenset[int]) -> str | None:
    if ns < 0:
        return None
    if ns in content_ids:
        return "content"
    if ns % 2 == 1:
        return "discussion"
    return None


def _labels(info: object) -> set[str]:
    if isinstance(info, str):
        return {info}
    if not isinstance(info, Mapping):
        return set()
    names: set[str] = set()
    if "canonical" in info and info["canonical"] is not None:
        names.add(str(info["canonical"]))
    if "*" in info and info["*"] is not None:
        names.add(str(info["*"]))
    return names


def groups_from_siteinfo(namespaces: Mapping) -> frozenset[int]:
    content: set[int] = set()
    for key, info in namespaces.items():
        if _labels(info) & CONTENT_NAMES:
            content.add(int(key))
    return frozenset(content)
