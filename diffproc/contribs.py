from diffproc.autoedits import matches_autoedit
from diffproc.config import PipelineConfig
from diffproc.namespaces import ns_group
from diffproc.types import Contrib, FilterDecision


def filter_contrib(
    contrib: Contrib,
    config: PipelineConfig | None = None,
    *,
    check_size: bool = True,
) -> FilterDecision:
    config = config or PipelineConfig()
    group = ns_group(contrib.ns, config.content_ids)
    if group is None:
        return FilterDecision(False, "excluded_namespace", None)

    if check_size and contrib.sizediff < config.thresholds_for(group).min_sizediff:
        return FilterDecision(False, "too_small", group)

    tags = contrib.tags
    if tags & config.blocked_tags or tags & config.tool_tags:
        return FilterDecision(False, "revert_or_tool", group)

    comment = contrib.comment or ""
    if matches_autoedit(contrib.ns, comment, tags, config.autoedits()):
        return FilterDecision(False, "automated", group)
    if config.copy_restore_pattern.search(comment):
        return FilterDecision(False, "copied_or_restored", group)
    if contrib.is_new and config.creation_copy_pattern.search(comment):
        return FilterDecision(False, "copied_or_restored", group)

    return FilterDecision(True, None, group)
