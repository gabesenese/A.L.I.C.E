import uvicorn

from app.bootstrap import create_app

app = create_app()


def __getattr__(name: str):
    if name == "ALICE":
        from app.legacy_main import ALICE

        return ALICE
    raise AttributeError(name)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
