# Cookiecutter — production-шаблон MLOps LLM

Полный стек **mlops-core**: inference, LoRA training, MLflow, MinIO, Admin/Chat UI, k3s-деплой.

## Быстрый старт

```bash
# из корня mlops-core
uv sync --group dev
uv run cookiecutter cookiecutter/ --no-input

# или интерактивно
uv run cookiecutter cookiecutter/
```

Сгенерированный проект появится в каталоге с именем `project_slug` (по умолчанию `my-mlops-llm`).

## Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `project_name` | Человекочитаемое имя | My MLOps LLM |
| `project_slug` | Slug для образов, pyproject, каталога | из `project_name` |
| `namespace` | Kubernetes namespace | `mlops` |
| `mlops_public_url` | Публичный URL (`http://IP`) | `http://localhost` |
| `public_host` | IP/хост без схемы | `localhost` |
| `lan_ip` | LAN IP для CORS | `192.168.0.103` |
| `mlops_registry` | Docker registry для k3s | `localhost:5000` |
| `auth_token_prefix` | Префикс JWT/localStorage | `mlops_` |
| `minio_bucket_*` | Имена S3-бакетов | `{slug}-models`, `{slug}-datasets` |
| `include_gpu` | GPU в app Deployment + CUDA deps | `yes` |
| `use_auth_service` | auth-service, login UI, JWT | `yes` |
| `use_monitoring` | Prometheus + Grafana | `yes` |
| `nodeport_*` | NodePort для UI/API | 30000–30901 |

## После генерации

```bash
cd <project_slug>
cp k8s/secrets.example.yaml k8s/secrets.yaml   # заполнить секреты
./scripts/k3s-build-images.sh
./scripts/k3s-deploy.sh
./scripts/k3s-copy-model.sh models/model.pt     # при необходимости
```

CPU-only: при `include_gpu=no` `k3s-deploy.sh` автоматически применяет `k8s/overlays/no-gpu/`.

## Поддержка шаблона (для maintainers)

После изменений в mlops-core синхронизируйте шаблон:

```bash
uv run python scripts/sync_cookiecutter_template.py
```

Скрипт копирует репозиторий в `cookiecutter/{{cookiecutter.project_slug}}/` и подставляет Jinja-переменные. Условная логика (GPU, auth, monitoring) — в `cookiecutter/overrides/` и `cookiecutter/hooks/`.

Проверка генерации:

```bash
uv run cookiecutter cookiecutter/ --no-input -o /tmp
ls /tmp/my-mlops-llm
kubectl kustomize /tmp/my-mlops-llm/k8s/ > /dev/null
```
