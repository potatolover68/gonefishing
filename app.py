import os

from factory import create_app


def main() -> None:
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        from encoder_service import ensure_encoder_server

        ensure_encoder_server()
    os.environ["GONEFISHING_REMOTE_ENCODER"] = "1"
    create_app().run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
else:
    app = create_app()
