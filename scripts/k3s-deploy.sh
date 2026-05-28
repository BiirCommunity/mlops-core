#!/usr/bin/env bash
# Деплой стека в k3s (namespace mlops).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUBLIC="${MLOPS_PUBLIC_URL:-http://83.221.210.29}"

if [[ ! -f k8s/secrets.yaml ]]; then
  echo "Создайте k8s/secrets.yaml из k8s/secrets.example.yaml и заполните секреты."
  exit 1
fi

echo "==> Applying secrets..."
kubectl apply -f k8s/secrets.yaml

echo "==> Applying manifests (kustomize)..."
kubectl apply -k k8s/

echo "==> Waiting for MinIO init job..."
kubectl -n mlops wait --for=condition=complete job/minio-init --timeout=180s || true

echo "==> Pods:"
kubectl -n mlops get pods

cat <<EOF

Стек развёрнут. NodePort (внешний IP ${PUBLIC#http://}):

  Admin UI       → ${PUBLIC}:30000
  Chat UI        → ${PUBLIC}:30100
  LLM API        → ${PUBLIC}:30800
  Grafana        → ${PUBLIC}:30300
  MLflow UI      → ${PUBLIC}:30500
  MinIO Console  → ${PUBLIC}:30901
  MinIO API (S3) → ${PUBLIC}:30900

Traefik (порт 80): ${PUBLIC}/admin , /chat , /api , /grafana , /mlflow

Bootstrap: admin / пароль из AUTH_BOOTSTRAP_PASSWORD в secrets.yaml
Grafana:   admin / GF_SECURITY_ADMIN_PASSWORD из secrets.yaml
MinIO:     MINIO_ACCESS_KEY / MINIO_SECRET_KEY из secrets.yaml

Проброс на роутере: 30000, 30100, 30300, 30500, 30800, 30900, 30901, 80

EOF
