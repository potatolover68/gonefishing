"""Keep the LUAR weights in one process so Flask debug restarts can reconnect."""

import os
import subprocess
import sys
import threading
import time
from multiprocessing.managers import BaseManager, RemoteError
from pathlib import Path

import numpy as np

from authorship.lookup import l2_normalize

ADDRESS = ("127.0.0.1", 8766)
AUTHKEY = b"gonefishing-encoder"
ROOT = Path(__file__).resolve().parent
REMOTE_BATCH = 16


class _EncoderProxy:
    def embed_episode(self, texts: list[str]) -> np.ndarray: ...

    def embed_many(self, episodes: list[list[str]]) -> np.ndarray: ...


class _EncoderManager(BaseManager):
    # BaseManager.register() installs this method at runtime. The body is only
    # here so type checkers can see the member it adds.
    def encoder(self) -> _EncoderProxy:
        raise AssertionError("encoder is created by BaseManager.register")


_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    _EncoderManager.register("encoder")
    _registered = True


def _connect():
    _ensure_registered()
    manager = _EncoderManager(address=ADDRESS, authkey=AUTHKEY)
    manager.connect()
    return manager


class RemoteLuarEncoder:
    def __init__(self) -> None:
        self._manager = None
        self._encoder = None
        self._connect_encoder()

    def _connect_encoder(self) -> None:
        try:
            self._manager = _connect()
            self._encoder = self._manager.encoder()
            return
        except (ConnectionRefusedError, EOFError, OSError, AttributeError, RemoteError):
            pass
        ensure_encoder_server()
        self._manager = _connect()
        self._encoder = self._manager.encoder()

    def _call(self, method: str, arg):
        try:
            return getattr(self._encoder, method)(arg)
        except (RemoteError, EOFError, ConnectionResetError, BrokenPipeError, OSError):
            self._connect_encoder()
            return getattr(self._encoder, method)(arg)

    def embed_episode(self, texts) -> np.ndarray:
        vectors = self.embed_many([[text] for text in texts])
        return l2_normalize(vectors.mean(axis=0))

    def embed_many(self, episodes) -> np.ndarray:
        episodes = list(episodes)
        rows = []
        for start in range(0, len(episodes), REMOTE_BATCH):
            chunk = episodes[start : start + REMOTE_BATCH]
            rows.append(np.asarray(self._call("embed_many", chunk)))
        if not rows:
            raise ValueError("expected at least one text")
        return np.concatenate(rows, axis=0)


def _register(encoder) -> None:
    global _registered
    _EncoderManager.register("encoder", callable=lambda: encoder)
    _registered = True


def serve() -> None:
    os.environ.setdefault("PYWIKIBOT_NO_USER_CONFIG", "2")
    from cpus import compute_cpus

    cores = str(compute_cpus())
    os.environ["OMP_NUM_THREADS"] = cores
    os.environ["MKL_NUM_THREADS"] = cores
    from authorship.encode import LuarEncoder, luar_onnx_path
    from scripts.collect_good_diffs import load_env

    env_path = ROOT / ".env"
    if env_path.exists():
        load_env(env_path)
    inner = LuarEncoder(
        luar_onnx_path(ROOT),
        ROOT / "data" / "luar_mud_lora" / "tokenizer.json",
    )
    lock = threading.Lock()

    class _Locked:
        def embed_episode(self, texts):
            with lock:
                return inner.embed_episode(texts)

        def embed_many(self, episodes):
            with lock:
                return inner.embed_many(episodes)

    encoder = _Locked()
    _register(encoder)
    manager = _EncoderManager(address=ADDRESS, authkey=AUTHKEY)
    server = manager.get_server()
    print("encoder ready", flush=True)
    server.serve_forever()


def ensure_encoder_server() -> None:
    try:
        _connect().encoder()
        return
    except (ConnectionRefusedError, EOFError, OSError, AttributeError):
        pass
    process = subprocess.Popen([sys.executable, str(ROOT / "encoder_service.py")])
    print("starting encoder", flush=True)

    def _stop() -> None:
        if process.poll() is None:
            process.terminate()

    import atexit

    atexit.register(_stop)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            _connect().encoder()
            return
        except (ConnectionRefusedError, EOFError, OSError, AttributeError):
            if process.poll() is not None:
                raise RuntimeError("encoder server exited before it was ready")
            time.sleep(0.25)
    raise RuntimeError("encoder server did not become ready")


if __name__ == "__main__":
    serve()
