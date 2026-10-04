"""Stored author prose and cached LUAR vectors."""

import sqlite3
from pathlib import Path

import numpy as np

from authorship.corpus import Document, load_documents


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


class AuthorStore:
    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "user TEXT NOT NULL, revid INTEGER PRIMARY KEY, pageid INTEGER NOT NULL, prose TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (user TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS similarities ("
            "query TEXT NOT NULL, other TEXT NOT NULL, score REAL NOT NULL, "
            "PRIMARY KEY (query, other))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tag_scores ("
            "query TEXT NOT NULL, tag TEXT NOT NULL, members TEXT NOT NULL, "
            "score REAL NOT NULL, PRIMARY KEY (query, tag))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tag_parts ("
            "tag TEXT NOT NULL, members TEXT NOT NULL, user TEXT NOT NULL, "
            "count INTEGER NOT NULL, sum BLOB NOT NULL, PRIMARY KEY (tag, user))"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def users(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT user FROM documents ORDER BY user"
        ).fetchall()
        return [row[0] for row in rows]

    def has_user(self, user: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE user = ? LIMIT 1", (user,)
        ).fetchone()
        return row is not None

    def count(self, user: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE user = ?", (user,)
        ).fetchone()
        return int(row[0])

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT user, COUNT(*) FROM documents GROUP BY user"
        ).fetchall()
        return {user: int(count) for user, count in rows}

    def prose(self, user: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT prose FROM documents WHERE user = ? ORDER BY revid", (user,)
        ).fetchall()
        return [row[0] for row in rows]

    def replace_user(self, documents: list[Document]) -> None:
        if not documents:
            return
        user = documents[0].user
        self._conn.execute("DELETE FROM documents WHERE user = ?", (user,))
        self._conn.execute("DELETE FROM vectors WHERE user = ?", (user,))
        self._conn.executemany(
            "INSERT OR REPLACE INTO documents (user, revid, pageid, prose) VALUES (?, ?, ?, ?)",
            [
                (document.user, document.revid, document.pageid, document.prose)
                for document in documents
            ],
        )
        self._conn.commit()

    def import_corpus(self, directory: Path) -> int:
        if self.users():
            return 0
        added = 0
        for author, documents in load_documents(directory).items():
            self.replace_user(documents)
            added += 1
        return added

    def vector(self, user: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT vector FROM vectors WHERE user = ?", (user,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def put_vector(self, user: str, vector: np.ndarray) -> None:
        blob = np.asarray(l2_normalize(vector), dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO vectors (user, vector) VALUES (?, ?)", (user, blob)
        )
        self._drop_scores(user)
        self._conn.commit()

    def scores_for(self, query: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT other, score FROM similarities WHERE query = ?", (query,)
        ).fetchall()
        return {other: float(score) for other, score in rows}

    def put_scores(self, query: str, scores: dict[str, float]) -> None:
        if not scores:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO similarities (query, other, score) VALUES (?, ?, ?)",
            [(query, other, score) for other, score in scores.items()],
        )
        self._conn.commit()

    def tag_score(self, query: str, tag: str, members: str) -> float | None:
        row = self._conn.execute(
            "SELECT score, members FROM tag_scores WHERE query = ? AND tag = ?",
            (query, tag),
        ).fetchone()
        if row is None or row[1] != members:
            return None
        return float(row[0])

    def put_tag_score(self, query: str, tag: str, members: str, score: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tag_scores (query, tag, members, score) VALUES (?, ?, ?, ?)",
            (query, tag, members, score),
        )
        self._conn.commit()

    def drop_tag_scores(self, tag: str) -> None:
        self._conn.execute("DELETE FROM tag_scores WHERE tag = ?", (tag,))
        self._conn.execute("DELETE FROM tag_parts WHERE tag = ?", (tag,))
        self._conn.commit()

    def tag_parts(self, tag: str, members: str) -> dict[str, tuple[np.ndarray, int]] | None:
        rows = self._conn.execute(
            "SELECT user, count, sum, members FROM tag_parts WHERE tag = ?", (tag,)
        ).fetchall()
        if not rows or any(row[3] != members for row in rows):
            return None
        return {
            row[0]: (np.frombuffer(row[2], dtype=np.float32).copy(), int(row[1]))
            for row in rows
        }

    def put_tag_parts(
        self, tag: str, members: str, parts: dict[str, tuple[np.ndarray, int]]
    ) -> None:
        self._conn.execute("DELETE FROM tag_parts WHERE tag = ?", (tag,))
        if parts:
            self._conn.executemany(
                "INSERT INTO tag_parts (tag, members, user, count, sum) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        tag,
                        members,
                        user,
                        count,
                        np.asarray(total, dtype=np.float32).tobytes(),
                    )
                    for user, (total, count) in parts.items()
                ],
            )
        self._conn.commit()

    def _drop_scores(self, user: str) -> None:
        self._conn.execute(
            "DELETE FROM similarities WHERE query = ? OR other = ?", (user, user)
        )
        needle = f"\n{user}\n"
        self._conn.execute(
            "DELETE FROM tag_scores WHERE query = ? OR instr(members, ?) > 0",
            (user, needle),
        )
        self._conn.execute(
            "DELETE FROM tag_parts WHERE user = ? OR instr(members, ?) > 0",
            (user, needle),
        )


def tag_similarity(
    query: np.ndarray, sums: list[np.ndarray], counts: list[int]
) -> float | None:
    total = None
    count = 0
    for vector, size in zip(sums, counts):
        if size <= 0:
            continue
        piece = np.asarray(vector, dtype=np.float64)
        total = piece if total is None else total + piece
        count += size
    if total is None or count <= 0:
        return None
    return float(np.dot(l2_normalize(np.asarray(query, dtype=np.float64)), l2_normalize(total)))


def centroid_distances(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    if not vectors:
        return {}
    names = list(vectors)
    stacked = np.vstack([np.asarray(vectors[name], dtype=np.float32) for name in names])
    centroid = l2_normalize(stacked.mean(axis=0))
    return {name: float(dot) for name, dot in zip(names, stacked @ centroid)}


def rank_authors(
    query: str, vectors: dict[str, np.ndarray], query_vector: np.ndarray
) -> list[tuple[str, float]]:
    scored = [
        (user, float(np.dot(query_vector, vector)))
        for user, vector in vectors.items()
        if user != query
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
