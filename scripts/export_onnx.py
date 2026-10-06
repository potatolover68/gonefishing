"""
requires pytorch... though there shouldn't be a real reason you're running this.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_good_diffs import load_env

MODEL_ID = "rrivera1849/LUAR-MUD"
ADAPTER = ROOT / "data" / "luar_mud_lora"
DEST = ROOT / "data" / "luar_mud.onnx"
TOKEN_LENGTH = 128
SAMPLES = (
    "The committee rejected the proposal after a long debate.",
    "See also the earlier discussion on the talk page.",
    "short",
    "A river of text " * 40,
)


class ExportableLUAR(nn.Module):
    """Episode forward without the Python document loop, so it can be traced."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.transformer = model.transformer
        self.linear = model.linear

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, episode, _ = input_ids.shape
        flat_ids = input_ids.reshape(batch * episode, TOKEN_LENGTH)
        flat_mask = attention_mask.reshape(batch * episode, TOKEN_LENGTH)
        hidden = self.transformer(
            input_ids=flat_ids, attention_mask=flat_mask
        ).last_hidden_state
        mask = flat_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        comments = (summed / counts).reshape(batch, episode, -1)
        scale = math.sqrt(comments.size(-1))
        scores = torch.matmul(comments, comments.transpose(-2, -1)) / scale
        attended = torch.matmul(torch.softmax(scores, dim=-1), comments)
        pooled = attended.max(dim=1).values
        return self.linear(pooled)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _require_close(name: str, left: np.ndarray, right: np.ndarray) -> None:
    score = min(_cosine(left[row], right[row]) for row in range(left.shape[0]))
    print(f"{name} cosine {score:.6f}", flush=True)
    if score <= 0.999:
        raise SystemExit(f"{name} drifted from PyTorch (cosine {score:.6f})")


def _reference(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, document_batch_size=32)
    return output.float().cpu().numpy()


def _tokenize(tokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        max_length=TOKEN_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def main() -> None:
    env = load_env(ROOT / ".env")
    token = env.get("HF_TOKEN") or None
    hub = {"token": token} if token else {}
    if token:
        import os

        os.environ["HF_TOKEN"] = token

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True, **hub)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, **hub)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, ADAPTER, **hub).merge_and_unload()
    model.to("cpu")
    model.eval()
    wrapper = ExportableLUAR(model).eval()

    ids, mask = _tokenize(tokenizer, list(SAMPLES))
    ids = ids.view(len(SAMPLES), 1, TOKEN_LENGTH)
    mask = mask.view(len(SAMPLES), 1, TOKEN_LENGTH)
    with torch.inference_mode():
        wrapped = wrapper(ids, mask).float().cpu().numpy()
    _require_close("wrapper", _reference(model, ids, mask), wrapped)

    generator = torch.Generator().manual_seed(0)
    wide_ids = torch.randint(0, 1000, (2, 16, TOKEN_LENGTH), generator=generator)
    wide_mask = torch.ones(2, 16, TOKEN_LENGTH, dtype=torch.long)
    wide_mask[:, :, 64:] = 0
    with torch.inference_mode():
        wide_wrapped = wrapper(wide_ids, wide_mask).float().cpu().numpy()
    _require_close("episode-16 wrapper", _reference(model, wide_ids, wide_mask), wide_wrapped)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (ids[:1], mask[:1]),
        str(DEST),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "episode"},
            "attention_mask": {0: "batch", 1: "episode"},
            "embedding": {0: "batch"},
        },
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(DEST), providers=["CPUExecutionProvider"])

    def run(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        return session.run(
            None,
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
            },
        )[0]

    _require_close("onnx strings", _reference(model, ids, mask), run(ids, mask))
    _require_close("onnx episode-16", _reference(model, wide_ids, wide_mask), run(wide_ids, wide_mask))
    print(f"wrote {DEST}", flush=True)


if __name__ == "__main__":
    main()
