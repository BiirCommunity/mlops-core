# Деплой в k3s через Argo CD

Образы собирает **GitHub Actions** → **GHCR** (`ghcr.io/biircommunity/mlops-core-*`).

## Быстрый старт

### 1. Argo CD

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side -f \
  https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

UI: `kubectl -n argocd port-forward svc/argocd-server 8080:443` → https://localhost:8080

### 2. Applications

```bash
kubectl apply -f deploy/argocd/application-secrets.yaml
kubectl apply -f deploy/argocd/application.yaml          # GPU
# kubectl apply -f deploy/argocd/application-no-gpu.yaml   # CPU
```

**mlops-core-secrets** → Edit → Helm → Parameters — пароли, затем **Sync**.  
**mlops-core** / **mlops-core-no-gpu** — **Sync**.

### 3. Модель

```bash
./scripts/k3s-copy-model.sh
```

## Теги образов (Argo CD UI)

Application → **Edit** → **Kustomize** → **Images**:

```yaml
kustomize:
  images:
    - mlops-core-app=ghcr.io/biircommunity/mlops-core-app:v1.0.0
    - mlops-core-auth-service=ghcr.io/biircommunity/mlops-core-auth-service:v1.0.0
    - mlops-core-admin-ui=ghcr.io/biircommunity/mlops-core-admin-ui:v1.0.0
    - mlops-core-chat-ui=ghcr.io/biircommunity/mlops-core-chat-ui:v1.0.0
    - mlops-core-mlflow=ghcr.io/biircommunity/mlops-core-mlflow:v1.0.0
```

Формат: `имя-из-kustomize=registry/образ:тег`. PR → тег = имя ветки, Release → тег релиза.

CLI:

```bash
argocd app set mlops-core \
  --kustomize-image mlops-core-app=ghcr.io/biircommunity/mlops-core-app:v1.0.0
argocd app sync mlops-core
```

## Секреты (Helm Application `mlops-core-secrets`)

| Parameter | Secret key |
|-----------|------------|
| `secrets.hfToken` | `HF_TOKEN` |
| `secrets.authBootstrapPassword` | `AUTH_BOOTSTRAP_PASSWORD` |
| `secrets.minioAccessKey` | `MINIO_ACCESS_KEY` |
| `secrets.minioSecretKey` | `MINIO_SECRET_KEY` |
| `secrets.grafanaAdminPassword` | `GF_SECURITY_ADMIN_PASSWORD` |

Остальные параметры: `deploy/helm/mlops-secrets/values.yaml`.

## NodePort

| Сервис | NodePort |
|--------|----------|
| admin-ui | 30000 |
| chat-ui | 30100 |
| grafana | 30300 |
| mlflow | 30500 |
| app (API) | 30800 |
| minio API | 30900 |
| minio console | 30901 |

Ingress Traefik: `/admin`, `/chat`, `/api`, `/mlflow`, `/grafana` (порт 80).

## Файлы

| Файл | Назначение |
|------|------------|
| `deploy/argocd/application-secrets.yaml` | Секреты (Helm) |
| `deploy/argocd/application.yaml` | Main app, GPU |
| `deploy/argocd/application-no-gpu.yaml` | Main app, CPU |
| `deploy/helm/mlops-secrets/` | Helm chart секретов |
| `k8s/` | Kustomize манифесты |
