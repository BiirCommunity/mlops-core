#!/usr/bin/env bash
# Деплой стека в k3s (namespace {{ cookiecutter.namespace }}).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUBLIC="${MLOPS_PUBLIC_URL:-{{ cookiecutter.mlops_public_url }}}"
{% if cookiecutter.include_gpu == "yes" %}
K8S_OVERLAY="k8s/"
{% else %}
K8S_OVERLAY="k8s/overlays/no-gpu/"
{% endif %}

if [[ ! -f k8s/secrets.yaml ]]; then
  echo "Создайте k8s/secrets.yaml из k8s/secrets.example.yaml и заполните секреты."
  exit 1
fi

echo "==> Applying secrets..."
kubectl apply -f k8s/secrets.yaml

echo "==> Applying manifests (kustomize: ${K8S_OVERLAY})..."
kubectl apply -k "${K8S_OVERLAY}"

echo "==> Waiting for MinIO init job..."
kubectl -n {{ cookiecutter.namespace }} wait --for=condition=complete job/minio-init --timeout=180s || true

echo "==> Pods:"
kubectl -n {{ cookiecutter.namespace }} get pods

cat <<EOF

Стек развёрнут. NodePort (внешний IP ${PUBLIC#http://}):

  Admin UI       → ${PUBLIC}:{{ cookiecutter.nodeport_admin }}
  Chat UI        → ${PUBLIC}:{{ cookiecutter.nodeport_chat }}
  LLM API        → ${PUBLIC}:{{ cookiecutter.nodeport_app }}
{% if cookiecutter.use_monitoring == "yes" %}
  Grafana        → ${PUBLIC}:{{ cookiecutter.nodeport_grafana }}
{% endif %}
  MLflow UI      → ${PUBLIC}:{{ cookiecutter.nodeport_mlflow }}
  MinIO Console  → ${PUBLIC}:{{ cookiecutter.nodeport_minio_console }}
  MinIO API (S3) → ${PUBLIC}:{{ cookiecutter.nodeport_minio_api }}

Traefik (порт 80): ${PUBLIC}/admin , /chat , /api , /mlflow{% if cookiecutter.use_monitoring == "yes" %} , /grafana{% endif %}

{% if cookiecutter.use_auth_service == "yes" %}
Bootstrap: admin / пароль из AUTH_BOOTSTRAP_PASSWORD в secrets.yaml
{% else %}
Inference API key: INFERENCE_API_KEY в secrets.yaml
{% endif %}
{% if cookiecutter.use_monitoring == "yes" %}
Grafana:   admin / GF_SECURITY_ADMIN_PASSWORD из secrets.yaml
{% endif %}
MinIO:     MINIO_ACCESS_KEY / MINIO_SECRET_KEY из secrets.yaml

Проброс на роутере: {{ cookiecutter.nodeport_admin }}, {{ cookiecutter.nodeport_chat }}{% if cookiecutter.use_monitoring == "yes" %}, {{ cookiecutter.nodeport_grafana }}{% endif %}, {{ cookiecutter.nodeport_mlflow }}, {{ cookiecutter.nodeport_app }}, {{ cookiecutter.nodeport_minio_api }}, {{ cookiecutter.nodeport_minio_console }}, 80

EOF
