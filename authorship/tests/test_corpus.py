import json

from authorship.corpus import joined_prose, load_documents


def test_joined_prose_skips_blanks():
    assert joined_prose({"passages": [{"prose": " One. "}, {"prose": "  "}, {"prose": "Two."}]}) == "One. Two."


def test_load_documents_uses_the_record_user(tmp_path):
    record = {
        "user": "Ada",
        "revid": 5,
        "pageid": 9,
        "passages": [{"prose": "Ada wrote this sentence."}],
    }
    (tmp_path / "someone.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    from_dir = load_documents(tmp_path)
    from_file = load_documents(tmp_path / "someone.jsonl")
    assert list(from_dir) == ["Ada"]
    assert from_dir["Ada"][0].pageid == 9
    assert from_file == from_dir
