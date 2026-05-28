# Деплой в k3s

## Предварительные условия

1. **k3s** установлен и `kubectl` настроен.
2. **GPU** (опционально): NVIDIA device plugin, иначе используйте overlay `k8s/overlays/no-gpu`.
3. **Локальный registry** настроен (`./scripts/k3s-setup-registry.sh`, см. [deploy/k3s/README.md](../deploy/k3s/README.md)).

## Шаги

### Секреты

```bash
cp k8s/secrets.example.yaml k8s/secrets.yaml
```

| Ключ | Назначение |
|------|------------|
| `HF_TOKEN` | HuggingFace (gated models) |
| `AUTH_BOOTSTRAP_PASSWORD` | Пароль admin при первом старте |
| `ACCESS_TOKEN` / `INFERENCE_API_KEY` | Legacy fallback, если auth-service недоступен |
| `MINIO_*` / `AWS_*` | MinIO credentials |
| `GF_SECURITY_ADMIN_*` | Grafana admin |

`k8s/secrets.yaml` в `.gitignore` — не коммитить.

### ConfigMap

Несекретные переменные — `k8s/configmap.yaml`. Редактируйте `TOKENIZER_NAME`, `CHECKPOINT_PATH`, MLflow buckets и т.д.

### Применение

```bash
kubectl apply -f k8s/secrets.yaml
kubectl apply -k k8s/
```

Или `./scripts/k3s-deploy.sh`.

### Сборка образов (build + push в registry)

```bash
./scripts/k3s-setup-registry.sh   # один раз
./scripts/k3s-build-images.sh     # docker build + push → {{ cookiecutter.mlops_registry }}
```

k3s подтягивает образы из registry при старте подов. После изменения кода — снова `k3s-build-images.sh`.

Не запускайте **microk8s** и **k3s** одновременно на одной машине — конфликт портов 10248/10250.

### PVC и модель

```bash
./scripts/k3s-copy-model.sh
```

Скрипт копирует `models/model.pt` в PVC через временный pod (работает даже если `app` в CrashLoop).

## NodePort

Публичный IP: `MLOPS_PUBLIC_URL` (по умолчанию `{{ cookiecutter.mlops_public_url }}`).

| Сервис | NodePort |
|--------|----------|
| admin-ui | 30000 |
| chat-ui | 30100 |
| grafana | 30300 |
| mlflow | 30500 |
| app (API) | 30800 |
| minio API | 30900 |
| minio console | 30901 |

## Ingress

Traefik (порт 80): пути `/admin`, `/chat`, `/api`, `/mlflow`, `/grafana` — работают и **без** Host (по IP).

Опционально с DNS: `mlops.local` → IP ноды.

## Без GPU

```bash
kubectl apply -k k8s/overlays/no-gpu/
```

Patch убирает `nvidia.com/gpu` и ставит `DEVICE=cpu` в ConfigMap.

## Обновление образов

```bash
./scripts/k3s-build-images.sh
```

## Мониторинг

```bash
kubectl -n {{ cookiecutter.namespace }} get pods
kubectl -n {{ cookiecutter.namespace }} logs -f deployment/app
kubectl -n {{ cookiecutter.namespace }} port-forward svc/grafana 4345:3000
```

## Миграция с Docker Compose

| Compose | k3s |
|---------|-----|
| `docker compose -f deploy/compose/docker-compose.yml up` | `kubectl apply -k k8s/` |
| `.env.docker.compose` | `secrets.yaml` + `configmap.yaml` |
| `./models` volume | PVC `models-pvc` |
| `redis-llm` | Service `redis` |
| порты 3000/3010/8080 | NodePort 30000/30100/30800 |

Compose-файл: [deploy/compose/docker-compose.yml](../deploy/compose/docker-compose.yml).
