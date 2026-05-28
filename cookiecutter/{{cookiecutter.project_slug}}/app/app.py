import os

from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI(title="{{ cookiecutter.project_name }}")
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    return {"status": "ok", "service": "{{ cookiecutter.project_slug }}"}


@app.get("/")
def root():
    return {
        "service": "{{ cookiecutter.project_slug }}",
        "docs": "/docs",
        "health": "/health",
        "mlflow": os.environ.get("MLFLOW_TRACKING_URI"),
    }
