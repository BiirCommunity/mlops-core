#!/usr/bin/env python3
"""Sync mlops-core into cookiecutter/{{cookiecutter.project_slug}}/ template.

Run from repo root after changing application code or k8s manifests:

    uv run python scripts/sync_cookiecutter_template.py
"""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cookiecutter" / "{{cookiecutter.project_slug}}"
OVERRIDES = ROOT / "cookiecutter" / "overrides"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "reports",
    "cookiecutter",
    ".idea",
    ".dvc/cache",
    ".dvc/tmp",
}

SKIP_REL_PATHS = {
    "k8s/secrets.yaml",
    ".env.docker.compose",
    "models/model.pt",
    "admin-ui/dist",
    "chat-ui/dist",
    "scripts/sync_cookiecutter_template.py",
}

# Order matters: longer / more specific tokens first.
REPLACEMENTS: list[tuple[str, str]] = [
    ("mlops-core-auth-service", "{{ cookiecutter.project_slug }}-auth-service"),
    ("mlops-core-admin-ui", "{{ cookiecutter.project_slug }}-admin-ui"),
    ("mlops-core-chat-ui", "{{ cookiecutter.project_slug }}-chat-ui"),
    ("mlops-core-mlflow", "{{ cookiecutter.project_slug }}-mlflow"),
    ("mlops-core-app", "{{ cookiecutter.project_slug }}-app"),
    ("mlops-config", "{{ cookiecutter.project_slug }}-config"),
    ("mlops-secrets", "{{ cookiecutter.project_slug }}-secrets"),
    ("mlops-ingress", "{{ cookiecutter.project_slug }}-ingress"),
    ("mlops-models", "{{ cookiecutter.minio_bucket_models }}"),
    ("mlops-datasets", "{{ cookiecutter.minio_bucket_datasets }}"),
    ("mlops_access_token", "{{ cookiecutter.auth_token_prefix }}access_token"),
    ("mlops_chat_token", "{{ cookiecutter.auth_token_prefix }}chat_token"),
    (
        'AUTH_TOKEN_PREFIX: "mlops_"',
        'AUTH_TOKEN_PREFIX: "{{ cookiecutter.auth_token_prefix }}"',
    ),
    (
        'token_prefix=os.environ.get("AUTH_TOKEN_PREFIX", "mlops_")',
        'token_prefix=os.environ.get("AUTH_TOKEN_PREFIX", "{{ cookiecutter.auth_token_prefix }}")',
    ),
    ("http://83.221.210.29", "{{ cookiecutter.mlops_public_url }}"),
    ("83.221.210.29", "{{ cookiecutter.public_host }}"),
    ("192.168.0.103", "{{ cookiecutter.lan_ip }}"),
    ("MLOps Core", "{{ cookiecutter.project_name }}"),
    ("mlops-core", "{{ cookiecutter.project_slug }}"),
    ("namespace: mlops", "namespace: {{ cookiecutter.namespace }}"),
    ("kubectl -n mlops", "kubectl -n {{ cookiecutter.namespace }}"),
    (
        "# Деплой стека в k3s (namespace mlops).",
        "# Деплой стека в k3s (namespace {{ cookiecutter.namespace }}).",
    ),
    (
        "storageClassName: local-path",
        "storageClassName: {{ cookiecutter.storage_class }}",
    ),
    ("host: mlops.local", "host: {{ cookiecutter.ingress_host }}"),
    (
        'MLFLOW_EXPERIMENT_NAME: "llm-lora-posttrain"',
        'MLFLOW_EXPERIMENT_NAME: "{{ cookiecutter.mlflow_experiment_name }}"',
    ),
    (
        'MLFLOW_MODEL_NAME: "llm-lora"',
        'MLFLOW_MODEL_NAME: "{{ cookiecutter.mlflow_model_name }}"',
    ),
    (
        'TOKENIZER_NAME: "meta-llama/Meta-Llama-3-8B"',
        'TOKENIZER_NAME: "{{ cookiecutter.tokenizer_name }}"',
    ),
    ("nodePort: 30000", "nodePort: {{ cookiecutter.nodeport_admin }}"),
    ("nodePort: 30100", "nodePort: {{ cookiecutter.nodeport_chat }}"),
    ("nodePort: 30300", "nodePort: {{ cookiecutter.nodeport_grafana }}"),
    ("nodePort: 30500", "nodePort: {{ cookiecutter.nodeport_mlflow }}"),
    ("nodePort: 30800", "nodePort: {{ cookiecutter.nodeport_app }}"),
    ("nodePort: 30900", "nodePort: {{ cookiecutter.nodeport_minio_api }}"),
    ("nodePort: 30901", "nodePort: {{ cookiecutter.nodeport_minio_console }}"),
    ("localhost:5000", "{{ cookiecutter.mlops_registry }}"),
    (':30500"}', ':{{ cookiecutter.nodeport_mlflow }}"}'),
    ("${PUBLIC}:30000", "${PUBLIC}:{{ cookiecutter.nodeport_admin }}"),
    ("${PUBLIC}:30100", "${PUBLIC}:{{ cookiecutter.nodeport_chat }}"),
    ("${PUBLIC}:30300", "${PUBLIC}:{{ cookiecutter.nodeport_grafana }}"),
    ("${PUBLIC}:30500", "${PUBLIC}:{{ cookiecutter.nodeport_mlflow }}"),
    ("${PUBLIC}:30800", "${PUBLIC}:{{ cookiecutter.nodeport_app }}"),
    ("${PUBLIC}:30900", "${PUBLIC}:{{ cookiecutter.nodeport_minio_api }}"),
    ("${PUBLIC}:30901", "${PUBLIC}:{{ cookiecutter.nodeport_minio_console }}"),
]

SKIP_RENDER_SUFFIXES = {
    ".json",
    ".pt",
    ".dvc",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".svg",
    ".lock",
    ".pyc",
    ".whl",
    ".gz",
    ".zip",
    ".bin",
}

SKIP_RENDER_NAMES = {"uv.lock", "package-lock.json"}


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not should_transform(src):
        shutil.copy2(src, dest)
        return
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dest)
        return
    dest.write_text(transform_text(text), encoding="utf-8")


def should_skip(rel: Path) -> bool:
    rel_posix = rel.as_posix()
    if rel_posix in SKIP_REL_PATHS:
        return True
    if rel_posix.startswith(".dvc/cache/") or rel_posix.startswith(".dvc/tmp/"):
        return True
    for part in rel.parts:
        if part in SKIP_DIR_NAMES:
            return True
    return False


def transform_text(content: str) -> str:
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)
    return content


def should_transform(path: Path) -> bool:
    if path.name in SKIP_RENDER_NAMES:
        return False
    return path.suffix not in SKIP_RENDER_SUFFIXES


def sync_tree() -> None:
    if TEMPLATE.exists():
        shutil.rmtree(TEMPLATE)
    TEMPLATE.mkdir(parents=True)

    for src in ROOT.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if should_skip(rel):
            continue
        dest = TEMPLATE / rel
        copy_file(src, dest)

    if OVERRIDES.is_dir():
        for src in OVERRIDES.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(OVERRIDES)
            dest = TEMPLATE / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    print(f"Synced template → {TEMPLATE.relative_to(ROOT)}")


if __name__ == "__main__":
    sync_tree()
