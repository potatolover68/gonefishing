import re
from dataclasses import dataclass, field
from pathlib import Path

from diffproc.autoedits import AutoEditRule, load_rules


@dataclass(frozen=True)
class GroupThresholds:
    min_sizediff: int
    min_prose_chars: int
    min_sentences: int
    min_prose_ratio: float


def _content_thresholds() -> GroupThresholds:
    return GroupThresholds(300, 300, 2, 0.5)


def _discussion_thresholds() -> GroupThresholds:
    return GroupThresholds(80, 100, 1, 0.4)


@dataclass(frozen=True)
class PipelineConfig:
    content_ids: frozenset[int] = field(default_factory=lambda: frozenset({0, 4, 118}))
    content: GroupThresholds = field(default_factory=_content_thresholds)
    discussion: GroupThresholds = field(default_factory=_discussion_thresholds)
    min_letter_ratio: float = 0.7
    min_words: int = 4
    containment_threshold: float = 0.7
    blocked_tags: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "mw-rollback",
                "mw-undo",
                "mw-manual-revert",
                "contenttranslation",
            }
        )
    )
    tool_tags: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "AWB",
                "twinkle",
                "huggle",
                "stiki",
                "redwarn",
                "ultraviolet",
            }
        )
    )
    copy_restore_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(
            r"\b(?:copied|split|merged)\s+from\b|\brestored\s+(?:revision|page|content)\b",
            re.IGNORECASE,
        )
    )
    creation_copy_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(
            r"\bmoved\s+from\b|\bsplit\s+from\b",
            re.IGNORECASE,
        )
    )
    interwiki_prefixes: frozenset[str] = field(
        default_factory=lambda: frozenset({"wikt", "commons", "m", "mw", "en"})
    )
    removed_tags: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"ref", "references", "gallery", "math", "syntaxhighlight", "table"}
        )
    )
    min_caps_letters: int = 20
    repeated_run_length: int = 10
    autoedit_rules: tuple[AutoEditRule, ...] | None = None
    autoedits_path: Path | None = None
    user_agent: str = ""
    maxlag: int = 5
    revision_batch_size: int = 50
    is_bot: bool = False

    def request_limit(self) -> int:
        if self.is_bot:
            return 500
        return self.revision_batch_size

    def autoedits(self) -> tuple[AutoEditRule, ...]:
        if self.autoedit_rules is not None:
            return self.autoedit_rules
        return load_rules(self.autoedits_path)

    def thresholds_for(self, ns_group: str) -> GroupThresholds:
        if ns_group == "content":
            return self.content
        if ns_group == "discussion":
            return self.discussion
        raise ValueError(f"unknown namespace group: {ns_group}")
