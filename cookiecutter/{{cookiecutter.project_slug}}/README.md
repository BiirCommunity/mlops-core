# {{ cookiecutter.project_name }}

Каркас MLOps-сервиса: FastAPI, k3s, MLflow, MinIO, опционально auth и Grafana.

## Локально

```bash
uv sync --group dev
uv run pytest
./run_app.sh
```

## k3s

```bash
cp k8s/secrets.example.yaml k8s/secrets.yaml
./scripts/k3s-build-images.sh
./scripts/k3s-deploy.sh
```

## Структура

```
app/                 заглушка API (/health, /metrics)
auth-service/        заглушка auth (если включён)
admin-ui/ chat-ui/   минимальные UI
k8s/ deploy/         манифесты и compose
scripts/             k3s и dvc
prometheus/          scrape app:8080
grafana-provisioning/
tests/
```

Дальше — замените `app/app.py` своей логикой.
