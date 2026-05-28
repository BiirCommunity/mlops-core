# Cookiecutter — каркас MLOps-проекта

Шаблон, не копия готового сервиса. На выходе — структура репозитория, инфра (k3s, MLflow, MinIO, CI) и минимальные заглушки app/UI.

## Новый проект

```bash
uv sync --group dev
uv run cookiecutter cookiecutter/
```

## Что внутри шаблона

| Каталог | Содержимое |
|---------|------------|
| `app/` | FastAPI: `/health`, `/metrics` |
| `auth-service/` | заглушка `/health` (если `use_auth_service=yes`) |
| `admin-ui/`, `chat-ui/` | одна страница, проверка API |
| `k8s/`, `deploy/` | Redis, MinIO, MLflow, app, UI, ingress |
| `scripts/` | k3s build/deploy, dvc-setup |
| `.github/` | black, pylint, docker build, kustomize |
| `prometheus/`, `grafana-provisioning/` | если `use_monitoring=yes` |

## Параметры

`project_slug`, `namespace`, `mlops_public_url`, `include_gpu`, `use_auth_service`, `use_monitoring`, nodeport'ы — в `cookiecutter.json`.

## Поддержка шаблона

Правки — только в `cookiecutter/{{cookiecutter.project_slug}}/`.

Корень репозитория `mlops-core` — **reference implementation** (полный LLM-стек), он не собирается из шаблона.

Проверка:

```bash
uv run cookiecutter cookiecutter/ --no-input -o /tmp
cd /tmp/my-mlops-project && uv sync --group dev && uv run pytest
```
