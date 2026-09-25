"""LUAR episode encoder."""

import os
from collections.abc import Sequence

import numpy as np
import torch

from authorship.lookup import l2_normalize

TOKEN_LENGTH = 128
GALLERY_CHUNK = 16


class LuarEncoder:
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        adapter: str | None = None,
        token: str | None = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if token:
            os.environ["HF_TOKEN"] = token
        hub = {"token": token} if token else {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_id = adapter or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, **hub)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **hub)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter, **hub)
        self.model.to(self.device)
        self.model.eval()

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
        encoded = self.tokenizer(
            flat,
            max_length=TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {
            key: value.view(len(episodes), length, -1).to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device.startswith("cuda")):
                output = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    document_batch_size=32,
                )
        return output.float().cpu().numpy()
