from diffproc.config import PipelineConfig
from diffproc.contribs import filter_contrib
from diffproc.extract import extract_passages
from diffproc.segment import diff_paragraphs
from diffproc.types import Contrib, ProcessResult
from diffproc.usefulness import check_usefulness


def process_edit(
    contrib: Contrib,
    old_wikitext: str | None,
    new_wikitext: str | None,
    config: PipelineConfig | None = None,
) -> ProcessResult:
    config = config or PipelineConfig()
    if new_wikitext is None or (old_wikitext is None and not contrib.is_new):
        return ProcessResult(False, "texthidden", (), False, None)

    decision = filter_contrib(contrib, config)
    if not decision.accept or decision.ns_group is None:
        return ProcessResult(False, decision.reason, (), False, None)

    old = "" if contrib.is_new else old_wikitext
    assert old is not None
    added, removed = diff_paragraphs(old, new_wikitext)
    if not added and not removed:
        useful, reason = check_usefulness((), decision.ns_group, config)
        return ProcessResult(True, None, (), useful, reason)

    passages = tuple(
        extract_passages(contrib, decision.ns_group, added, removed, config)
    )
    useful, reason = check_usefulness(passages, decision.ns_group, config)
    return ProcessResult(True, None, passages, useful, reason)
