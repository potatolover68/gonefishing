"""Load per-author edit documents from data/good_diffs."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    user: str
    revid: int
    pageid: int
    prose: str


def joined_prose(record: dict) -> str:
    parts = []
    for passage in record.get("passages") or []:
        text = (passage.get("prose") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def load_documents(path: Path) -> dict[str, list[Document]]:
    by_author: dict[str, list[Document]] = defaultdict(list)
    seen: set[int] = set()
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for file in paths:
        for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping malformed {file.name}:{line_number}", flush=True)
                continue
            prose = joined_prose(record)
            if not prose:
                continue
            revid = int(record["revid"])
            if revid in seen:
                continue
            seen.add(revid)
            document = Document(
                user=str(record["user"]),
                revid=revid,
                pageid=int(record["pageid"]),
                prose=prose,
            )
            by_author[document.user].append(document)
    return dict(by_author)
