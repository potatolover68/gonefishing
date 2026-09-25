from authorship.corpus import Document
from authorship.lookup import AuthorStore, rank_authors

import numpy as np


def test_store_roundtrip_and_rank(tmp_path):
    store = AuthorStore(tmp_path / "authors.sqlite")
    store.replace_user(
        [
            Document("Ada", 1, 10, "Ada wrote this sentence."),
            Document("Bea", 2, 11, "Bea wrote another sentence."),
        ]
    )
    assert store.has_user("Ada")
    assert store.prose("Ada") == ["Ada wrote this sentence."]
    assert store.counts() == {"Ada": 1, "Bea": 1}
    ada = np.array([1.0, 0.0], dtype=np.float32)
    bea = np.array([0.0, 1.0], dtype=np.float32)
    store.put_vector("Ada", ada)
    store.put_vector("Bea", bea)
    ranked = rank_authors("Ada", {"Ada": store.vector("Ada"), "Bea": store.vector("Bea")}, store.vector("Ada"))
    assert [name for name, _score in ranked] == ["Bea"]
    store.put_scores("Ada", {"Bea": 0.5})
    assert store.scores_for("Ada")["Bea"] == 0.5
    store.put_tag_score("Ada", "socks", "\nBea\n", 0.25)
    assert store.tag_score("Ada", "socks", "\nBea\n") == 0.25
    store.put_vector("Bea", bea)
    assert store.scores_for("Ada") == {}
    assert store.tag_score("Ada", "socks", "\nBea\n") is None
    store.close()
