"""Keep the LUAR weights in one process so Flask debug restarts can reconnect."""

import os
import subprocess
import sys
import threading
import time
from multiprocessing.managers import BaseManager
from pathlib import Path

import numpy as np

ADDRESS = ("127.0.0.1", 8766)
AUTHKEY = b"gonefishing-encoder"
ROOT = Path(__file__).resolve().parent


class _EncoderProxy:
    def embed_episode(self, texts: list[str]) -> np.ndarray: ...


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
        self._manager = _connect()
        self._encoder = self._manager.encoder()

    def embed_episode(self, texts) -> np.ndarray:
        return np.asarray(self._encoder.embed_episode(list(texts)))


def _register(encoder) -> None:
    global _registered
    _EncoderManager.register("encoder", callable=lambda: encoder)
    _registered = True


def serve() -> None:
    os.environ.setdefault("PYWIKIBOT_NO_USER_CONFIG", "2")
    from authorship.encode import LuarEncoder
    from scripts.collect_good_diffs import load_env

    env_path = ROOT / ".env"
    env = load_env(env_path) if env_path.exists() else {}
    inner = LuarEncoder(
        "rrivera1849/LUAR-MUD",
        adapter=str(ROOT / "data" / "luar_mud_lora"),
        token=env.get("HF_TOKEN") or None,
    )
    lock = threading.Lock()

    class _Locked:
        def embed_episode(self, texts):
            with lock:
                return inner.embed_episode(texts)

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
