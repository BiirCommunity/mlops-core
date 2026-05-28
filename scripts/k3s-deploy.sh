#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUBLIC="${MLOPS_PUBLIC_URL:-http://83.221.210.29}"

if [[ ! -f k8s/secrets.yaml ]]; then
  echo "Нужен k8s/secrets.yaml — скопируй из k8s/secrets.example.yaml и заполни."
  exit 1
fi

kubectl apply -f k8s/secrets.yaml
kubectl apply -k k8s/

kubectl -n mlops wait --for=condition=complete job/minio-init --timeout=180s || true
kubectl -n mlops get pods

cat <<EOF

Готово. NodePort на ${PUBLIC#http://}:

  Admin UI       ${PUBLIC}:30000
  Chat UI        ${PUBLIC}:30100
  LLM API        ${PUBLIC}:30800
  Grafana        ${PUBLIC}:30300
  MLflow         ${PUBLIC}:30500
  MinIO Console  ${PUBLIC}:30901
  MinIO S3       ${PUBLIC}:30900

Через Traefik (80): ${PUBLIC}/admin /chat /api /grafana /mlflow

Логины:
  admin UI — admin + AUTH_BOOTSTRAP_PASSWORD из secrets.yaml
  Grafana  — admin + GF_SECURITY_ADMIN_PASSWORD
  MinIO    — MINIO_ACCESS_KEY / MINIO_SECRET_KEY

На роутере пробрось: 30000 30100 30300 30500 30800 30900 30901 80

EOF
