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

Формат: полный путь образа после `components/ghcr-images`, например `ghcr.io/org/mlops-core-app:v1.0.0` (короткое имя `mlops-core-app=…` не сработает). PR → тег = имя ветки, Release → тег релиза.

CLI:

```bash
argocd app patch mlops-core --type merge --patch '{
  "spec": {"source": {"kustomize": {"images": [
    "ghcr.io/biircommunity/mlops-core-app:v1.0.0"
  ]}}}}
}'
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

## Ingress, TLS (свои сертификаты) и GitHub webhook

### Домен — в Helm-параметрах (как секреты)

Ingress **не умеет** читать host из Secret напрямую. Домен задаётся в Application **mlops-core-secrets** → Edit → Helm → Parameters:

| Parameter | Пример | Назначение |
|-----------|--------|------------|
| `ingress.domain` | `adaptive-llm.ru` | Host в Ingress mlops и Argo CD |
| `ingress.tlsSecretName` | `adaptive-llm-tls` | Имя Secret с `tls.crt` / `tls.key` |
| `ingress.enabled` | `true` | Создавать Ingress |

Файл по умолчанию: `deploy/helm/mlops-secrets/values.yaml`.

Helm генерирует:
- Ingress для каждого path + Traefik Middleware (strip/rewrite)
- Ingress `argocd-server` (namespace `argocd`)
- ConfigMap `mlops-platform-config` (`DOMAIN`, `GRAFANA_ROOT_URL`, `ADMIN_UI_ORIGINS`, …)

### Path routing

| URL | Как работает |
|-----|----------------|
| `/admin`, `/chat` | Traefik strip prefix + Vite `base` + React `basename` |
| `/api/*` | Traefik rewrite → `app` (`/training/*`, `/v1/*`) или `auth-service` |
| `/grafana` | `GF_SERVER_SERVE_FROM_SUB_PATH` + `GF_SERVER_ROOT_URL` |
| `/mlflow` | `MLFLOW_STATIC_PREFIX=/mlflow` |
| `/minio` | Traefik strip prefix + `MINIO_BROWSER_REDIRECT_URL` (console) |
| `/minio-api` | Traefik strip prefix + `MINIO_PUBLIC_API_URL` (внешний S3) |
| `/minio` | `MINIO_BROWSER_REDIRECT_URL`; `MINIO_SERVER_URL` = `http://minio:9000` (внутренний API для Console login) |
| `/argocd` | `server.rootpath` в Argo CD |

После изменений UI — пересобрать `mlops-core-admin-ui` и `mlops-core-chat-ui`, Sync `mlops-core-secrets` и `mlops-core`.

### Один домен для всего

| URL | Сервис |
|-----|--------|
| `https://<domain>/admin` | admin-ui |
| `https://<domain>/chat` | chat-ui |
| `https://<domain>/api` | API |
| `https://<domain>/mlflow` | MLflow |
| `https://<domain>/minio` | MinIO Console |
| `https://<domain>/minio-api` | MinIO S3 API |
| `https://<domain>/argocd` | Argo CD UI |
| `https://<domain>/argocd/api/webhook` | GitHub webhook |

### Argo CD UI на `/argocd`

```bash
kubectl apply -f deploy/argocd/server-insecure.yaml
kubectl apply -f deploy/argocd/argocd-server-subpath.yaml
kubectl -n argocd rollout status deployment/argocd-server
```

Нужны **оба** `server.basehref` и `server.rootpath` — без `basehref` UI белый экран (JS грузится с `/` вместо `/argocd/`).

URL: `https://<domain>/argocd/` (со слэшем в конце).

### TLS-сертификат (файлы, не в Git)

Secret с ключами создаётся вручную **в обоих namespace** (имя = `ingress.tlsSecretName`):

```bash
kubectl -n mlops create secret tls adaptive-llm-tls --cert=fullchain.pem --key=privkey.pem
kubectl -n argocd create secret tls adaptive-llm-tls --cert=fullchain.pem --key=privkey.pem
```

Сами `.pem` в Git/Helm values не кладём — только имя secret.

### Первый запуск

```bash
kubectl apply -f deploy/argocd/server-insecure.yaml
kubectl apply -f deploy/argocd/application-secrets.yaml
# TLS secrets + параметры ingress.domain в UI → Sync mlops-core-secrets
```

### GitHub → Argo CD (webhook)

GitHub должен достучаться до `https://<argocd-домен>/api/webhook` (публичный IP, проброс 443 или туннель).

1. GitHub → **Settings** → **Webhooks** → **Add webhook**
2. **Payload URL:** `https://<ingress.domain>/argocd/api/webhook` (или `WEBHOOK_URL` из ConfigMap `mlops-platform-config`)
3. **Content type:** `application/json`
4. **Events:** Push

Опционально secret в Argo CD:

```bash
kubectl -n argocd patch secret argocd-secret \
  -p '{"stringData":{"webhook.github.secret":"ВАШ_СЕКРЕТ"}}'
```

Auto-sync уже в `deploy/argocd/application.yaml`. Для приватного репо — credentials в Argo CD → Settings → Repositories.

### Release → Argo CD (автодеплой по tag)

Workflow **Release** после push образов в GHCR вызывает Argo CD API и обновляет Application **mlops-core**.

**GitHub → Settings → Secrets and variables → Actions → New repository secret:**

| Secret | Значение |
|--------|----------|
| `ARGOCD_AUTH_TOKEN` | токен из Argo CD UI (Settings → Accounts → Generate token) |

**GitHub → Settings → Environments → New environment** → имя **`production`** (как в `release.yml`):

1. **Required reviewers** — один или несколько approvers (деплой в Argo CD не стартует без Approve).
2. Опционально: **Wait timer**, **Deployment branches** (только `main`).

После `publish` job `deploy-argo` ждёт approval в Actions → run → **Review pending deployments** → **Approve and deploy**.  
`lint` / `build` / `publish` идут без паузы; пауза только перед patch/sync в Argo CD.

Argo CD должен быть доступен из GitHub Actions: `https://adaptive-llm.ru/argocd`.

Запуск: **Actions → Release → Run workflow** → tag, например `v1.2.0`.

Job `deploy-argo`:
1. `argocd app patch mlops-core --type merge` — полный список `spec.source.kustomize.images` (формат `ghcr.io/…/image:tag`, без дублей)
2. `argocd app sync mlops-core --prune`
3. `app wait --sync`, затем `--health` для UI/API, отдельно `app` (GPU, timeout 40 мин)

Переменные в `.github/workflows/release.yml`: `ARGOCD_SERVER`, `ARGOCD_ROOT_PATH`, `ARGOCD_APP_NAME`.

Job `minio-init` после завершения удаляется TTL-controller'ом — в `application.yaml` задан `ignoreDifferences` для `/status`, иначе Application остаётся OutOfSync.

Для CPU-кластера замените `ARGOCD_APP_NAME` на `mlops-core-no-gpu` или добавьте второй job.

### Let's Encrypt (опционально)

Если позже понадобится автоматический cert — `deploy/ingress/cluster-issuer.yaml` + cert-manager. Для своих CA не используйте.

### Файлы ingress/TLS

| Файл | Назначение |
|------|------------|
| `deploy/argocd/server-insecure.yaml` | TLS на Traefik, `server.rootpath` |
| `deploy/helm/mlops-secrets/` | Секреты + **ingress.domain** + Ingress |
| `deploy/ingress/cluster-issuer.yaml` | Только Let's Encrypt (опционально) |

## Файлы

| Файл | Назначение |
|------|------------|
| `deploy/argocd/application-secrets.yaml` | Секреты (Helm) |
| `deploy/argocd/application.yaml` | Main app, GPU |
| `deploy/argocd/application-no-gpu.yaml` | Main app, CPU |
| `deploy/helm/mlops-secrets/` | Helm: секреты, домен, Ingress |
| `k8s/` | Kustomize манифесты |
