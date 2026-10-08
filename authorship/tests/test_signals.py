import numpy as np

from authorship.signals import (
    AUTHORSHIP_WEIGHT,
    TOPIC_CLOSE,
    TOPIC_WEIGHT,
    EditFact,
    average_topics,
    blend_similarity,
    hour_histogram,
    topic_sample,
    topics_from_payload,
)


def _fact(revid: int, ns: int = 0, sizediff: int = 500, automated: bool = False, hour: int = 3) -> EditFact:
    return EditFact(
        user="Ada",
        revid=revid,
        pageid=revid,
        ns=ns,
        title=f"Page {revid}",
        timestamp=f"2026-01-05T{hour:02d}:00:00Z",
        sizediff=sizediff,
        comment="",
        tags=(),
        automated=automated,
    )


def test_topic_sample_keeps_the_largest_mainspace_edits():
    facts = [_fact(index, sizediff=100 + index) for index in range(20)]
    facts.append(_fact(100, ns=1, sizediff=9000))
    facts.append(_fact(101, sizediff=50))
    facts.append(_fact(102, sizediff=8000, automated=True))
    chosen = topic_sample(facts)
    assert len(chosen) == 10
    assert len(topic_sample([_fact(index, sizediff=200) for index in range(250)])) == 25
    assert chosen[0].revid == 19
    assert all(fact.ns == 0 and not fact.automated for fact in chosen)
    assert all(abs(fact.sizediff) >= 100 for fact in chosen)


def test_topic_sample_can_be_shorter_than_ten():
    facts = [_fact(1, sizediff=150), _fact(2, sizediff=40), _fact(3, ns=4, sizediff=900)]
    assert [fact.revid for fact in topic_sample(facts)] == [1]


def test_hour_histogram_skips_automated_edits_and_weights_bytes():
    facts = [
        _fact(1, sizediff=10, hour=4),
        _fact(2, sizediff=-30, hour=4),
        _fact(3, sizediff=100, hour=4, automated=True),
        _fact(4, sizediff=5, hour=1),
    ]
    bins = hour_histogram(facts)
    assert bins is not None
    assert bins[4] == 40
    assert bins[1] == 5
    assert bins.sum() == 45


def test_time_of_day_cannot_lower_the_score():
    low, low_detail = blend_similarity(0.8, None, 0.1)
    assert abs(low - 0.8) < 1e-9
    assert low_detail == "authorship 0.8000, time 0.1000"
    high, high_detail = blend_similarity(0.5, None, 1.0)
    assert high > 0.5
    assert "time 1.0000" in high_detail
    score, detail = blend_similarity(0.5, None, None)
    assert score == 0.5
    assert detail == "authorship 0.5000"
    mixed, mixed_detail = blend_similarity(0.5, 1.0, None)
    assert mixed == (AUTHORSHIP_WEIGHT * 0.5 + TOPIC_WEIGHT * 1.0) / (
        AUTHORSHIP_WEIGHT + TOPIC_WEIGHT
    )
    assert mixed_detail == "authorship 0.5000, topic 1.0000"
    loose, loose_detail = blend_similarity(0.8, TOPIC_CLOSE - 0.01, None)
    assert abs(loose - 0.8) < 1e-9
    assert "topic" in loose_detail


def test_average_topics_and_payload():
    averaged = average_topics([{"Culture": 1.0, "STEM": 0.0}, {"Culture": 0.0, "STEM": 1.0}])
    assert averaged["Culture"] == 0.5
    assert averaged["STEM"] == 0.5
    parsed = topics_from_payload(
        {"prediction": {"results": [{"topic": "Culture", "score": 0.25}]}}
    )
    assert parsed == {"Culture": 0.25}
    assert hour_histogram([_fact(1, automated=True)]) is None
    assert np.allclose(
        hour_histogram([_fact(1, sizediff=2, hour=0)]),
        np.array([2] + [0] * 23),
    )
