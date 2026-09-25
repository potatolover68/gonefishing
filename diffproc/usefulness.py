import re

from diffproc.config import PipelineConfig
from diffproc.types import Passage


def check_usefulness(
    passages: list[Passage] | tuple[Passage, ...],
    ns_group: str,
    config: PipelineConfig | None = None,
) -> tuple[bool, str | None]:
    config = config or PipelineConfig()
    thresholds = config.thresholds_for(ns_group)
    fresh = [passage for passage in passages if passage.kind == "new"]
    prose = "\n".join(passage.prose for passage in fresh)
    n_chars = sum(passage.n_chars for passage in fresh)
    n_sentences = sum(passage.n_sentences for passage in fresh)
    if fresh:
        inserted = fresh[0].inserted_chars_total
    elif passages:
        inserted = passages[0].inserted_chars_total
    else:
        inserted = 0

    if n_chars < thresholds.min_prose_chars:
        return False, "too_little_prose"
    if n_sentences < thresholds.min_sentences:
        return False, "too_few_sentences"
    if inserted <= 0 or n_chars / inserted < thresholds.min_prose_ratio:
        return False, "low_prose_ratio"

    letters = [char for char in prose if char.isalpha()]
    if n_chars == 0 or len(letters) / n_chars < config.min_letter_ratio:
        return False, "low_letter_ratio"
    if len(letters) >= config.min_caps_letters and all(
        char.isupper() for char in letters
    ):
        return False, "all_caps"
    run = config.repeated_run_length - 1
    if re.search(rf"(.)\1{{{run},}}", prose):
        return False, "repeated_run"
    return True, None
