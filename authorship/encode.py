"""LUAR episode encoder."""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from authorship.lookup import l2_normalize
from cpus import compute_cpus

TOKEN_LENGTH = 128
GALLERY_CHUNK = 16


def luar_onnx_path(root: Path) -> Path:
    if os.environ.get("ON_TF"):
        home = os.environ.get("TOOL_DATA_DIR") or os.environ.get("HOME")
        if not home:
            raise RuntimeError("ON_TF is set but HOME is not")
        return Path(home) / "gonefishing" / "luar_mud.onnx"
    return root / "data" / "luar_mud.onnx"


class LuarEncoder:
    def __init__(self, onnx_path: Path, tokenizer_path: Path) -> None:
        import onnxruntime as ort

        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"LUAR ONNX file not found at {path}. "
                "Run scripts/export_onnx.py and copy the file there."
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = compute_cpus()
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def embed_episode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("expected at least one text")
        vectors = self.embed_many([[text] for text in texts])
        return l2_normalize(vectors.mean(axis=0))

    def embed_many(self, episodes: list[Sequence[str]]) -> np.ndarray:
        if len(episodes) == 1 and len(episodes[0]) > GALLERY_CHUNK:
            chunks = [
                list(episodes[0][start : start + GALLERY_CHUNK])
                for start in range(0, len(episodes[0]), GALLERY_CHUNK)
            ]
            vectors = [self._forward([chunk])[0] for chunk in chunks]
            return l2_normalize(np.mean(np.stack(vectors), axis=0))[None, :]
        rows = []
        for start in range(0, len(episodes), 16):
            rows.append(self._forward(list(episodes[start : start + 16])))
        return np.concatenate(rows, axis=0)

    def _forward(self, episodes: list[Sequence[str]]) -> np.ndarray:
        length = len(episodes[0])
        if any(len(episode) != length for episode in episodes):
            raise ValueError("LUAR episodes in one forward must share a length")
        flat = [text for episode in episodes for text in episode]
        encoded = self._tokenizer.encode_batch(flat)
        ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        ids = ids.reshape(len(episodes), length, TOKEN_LENGTH)
        mask = mask.reshape(len(episodes), length, TOKEN_LENGTH)
        output = self._session.run(
            None, {"input_ids": ids, "attention_mask": mask}
        )[0]
        return np.asarray(output, dtype=np.float32)
