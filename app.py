import os

from factory import create_app


def main() -> None:
    os.environ["GONEFISHING_REMOTE_ENCODER"] = "1"
    create_app().run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
else:
    os.environ["GONEFISHING_REMOTE_ENCODER"] = "1"
    app = create_app(init_heavy=False)
