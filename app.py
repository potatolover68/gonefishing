import os

from factory import create_app


def _build_app():
    os.environ["GONEFISHING_REMOTE_ENCODER"] = "1"
    from encoder_service import ensure_encoder_server

    ensure_encoder_server()
    return create_app()


def main() -> None:
    _build_app().run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
else:
    app = _build_app()
