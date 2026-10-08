"""Topic and time-of-day signals stored beside an authorship vector."""

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from diffproc.autoedits import matches_autoedit
from diffproc.config import PipelineConfig
from diffproc.types import Contrib

TOPIC_URL = (
    "https://api.wikimedia.org/service/lw/inference/v1/models/"
    "outlink-topic-model:predict"
)
USER_AGENT = (
    "gonefishing/0.1 by en:User:MSK <thewonderfulworldofpotatoes at gmail dot com>"
)
MIN_TOPIC_BYTES = 100
AUTHORSHIP_WEIGHT = 0.75
TOPIC_WEIGHT = 0.10
TIME_WEIGHT = 0.15
TOPIC_CLOSE = 0.9


@dataclass(frozen=True)
class EditFact:
    user: str
    revid: int
    pageid: int
    ns: int
    title: str
    timestamp: str
    sizediff: int
    comment: str
    tags: tuple[str, ...]
    automated: bool


def fact_from_contrib(contrib: Contrib, config: PipelineConfig) -> EditFact:
    automated = (
        matches_autoedit(
            contrib.ns, contrib.comment or "", contrib.tags, config.autoedits()
        )
        is not None
    )
    return EditFact(
        user=contrib.user,
        revid=contrib.revid,
        pageid=contrib.pageid,
        ns=contrib.ns,
        title=contrib.title,
        timestamp=contrib.timestamp,
        sizediff=contrib.sizediff,
        comment=contrib.comment or "",
        tags=tuple(sorted(contrib.tags)),
        automated=automated,
    )


def topic_sample(facts: list[EditFact]) -> list[EditFact]:
    eligible = [
        fact
        for fact in facts
        if fact.ns == 0 and not fact.automated and abs(fact.sizediff) >= MIN_TOPIC_BYTES
    ]
    if not eligible:
        return []
    count = max(10, math.ceil(len(eligible) * 0.1))
    eligible.sort(key=lambda fact: abs(fact.sizediff), reverse=True)
    return eligible[:count]


def hour_histogram(facts: list[EditFact]) -> np.ndarray | None:
    bins = np.zeros(24, dtype=np.float64)
    for fact in facts:
        if fact.automated:
            continue
        parsed = _hour(fact.timestamp)
        if parsed is None:
            continue
        bins[parsed] += abs(fact.sizediff)
    if float(bins.sum()) == 0:
        return None
    return bins


def cosine_dicts(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> float | None:
    if not left or not right:
        return None
    keys = sorted(set(left) | set(right))
    return _cosine(
        np.asarray([left.get(key, 0.0) for key in keys], dtype=np.float64),
        np.asarray([right.get(key, 0.0) for key in keys], dtype=np.float64),
    )


def cosine_histograms(
    left: np.ndarray | None, right: np.ndarray | None
) -> float | None:
    if left is None or right is None:
        return None
    return _cosine(left, right)


def blend_similarity(
    authorship: float | None,
    topic: float | None,
    time_of_day: float | None,
) -> tuple[float | None, str]:
    """Authorship is the score. Time and a very close topic can only raise it."""
    if authorship is None and topic is None and time_of_day is None:
        return None, ""
    if authorship is None:
        authorship = 0.0
    score = authorship
    total = AUTHORSHIP_WEIGHT
    detail = [f"authorship {authorship:.4f}"]
    if time_of_day is not None:
        detail.append(f"time {time_of_day:.4f}")
        boosted = (score * total + TIME_WEIGHT * time_of_day) / (total + TIME_WEIGHT)
        if boosted > score:
            score = boosted
            total += TIME_WEIGHT
    if topic is not None:
        detail.append(f"topic {topic:.4f}")
        if topic >= TOPIC_CLOSE:
            boosted = (score * total + TOPIC_WEIGHT * topic) / (total + TOPIC_WEIGHT)
            if boosted > score:
                score = boosted
    return score, ", ".join(detail)


def average_topics(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    totals: dict[str, float] = {}
    for sample in samples:
        for topic, score in sample.items():
            totals[topic] = totals.get(topic, 0.0) + score
    count = len(samples)
    return {topic: total / count for topic, total in totals.items()}


def topics_from_payload(payload: dict) -> dict[str, float]:
    prediction = (
        payload.get("prediction")
        if isinstance(payload.get("prediction"), dict)
        else payload
    )
    results = prediction.get("results") if isinstance(prediction, dict) else None
    scores: dict[str, float] = {}
    for item in results or []:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        score = item.get("score")
        if topic is None or score is None:
            continue
        scores[str(topic)] = float(score)
    return scores


def predict_topics(title: str, revid: int) -> dict[str, float]:
    body = json.dumps(
        {
            "lang": "en",
            "page_title": title.replace(" ", "_"),
            "revision_id": revid,
            "threshold": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        TOPIC_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        return {}
    return topics_from_payload(payload)


def collect_topic_vector(facts: list[EditFact]) -> dict[str, float] | None:
    sample = topic_sample(facts)
    if not sample:
        return {}
    samples: list[dict[str, float]] = []
    for fact in sample:
        try:
            scores = predict_topics(fact.title, fact.revid)
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            print(f"topic {fact.revid} failed: {error}", flush=True)
            continue
        if scores:
            samples.append(scores)
    if not samples:
        return None
    return average_topics(samples)


def _hour(timestamp: str) -> int | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).hour


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))
