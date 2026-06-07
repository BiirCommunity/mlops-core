from fastapi import FastAPI

app = FastAPI(title="{{ cookiecutter.project_slug }}-auth")


@app.get("/health")
def health():
    return {"status": "ok"}
