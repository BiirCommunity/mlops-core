#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUBLIC="${MLOPS_PUBLIC_URL:-{{ cookiecutter.mlops_public_url }}}"
{% if cookiecutter.include_gpu == "yes" %}
K8S_OVERLAY="k8s/"
{% else %}
K8S_OVERLAY="k8s/overlays/no-gpu/"
{% endif %}

if [[ ! -f k8s/secrets.yaml ]]; then
  echo "Нужен k8s/secrets.yaml — скопируй из k8s/secrets.example.yaml и заполни."
  exit 1
fi

kubectl apply -f k8s/secrets.yaml
kubectl apply -k "${K8S_OVERLAY}"

kubectl -n {{ cookiecutter.namespace }} wait --for=condition=complete job/minio-init --timeout=180s || true
kubectl -n {{ cookiecutter.namespace }} get pods

cat <<EOF

Готово. NodePort на ${PUBLIC#http://}:

  Admin UI       ${PUBLIC}:{{ cookiecutter.nodeport_admin }}
  Chat UI        ${PUBLIC}:{{ cookiecutter.nodeport_chat }}
  LLM API        ${PUBLIC}:{{ cookiecutter.nodeport_app }}
{% if cookiecutter.use_monitoring == "yes" %}
  Grafana        ${PUBLIC}:{{ cookiecutter.nodeport_grafana }}
{% endif %}
  MLflow         ${PUBLIC}:{{ cookiecutter.nodeport_mlflow }}
  MinIO Console  ${PUBLIC}:{{ cookiecutter.nodeport_minio_console }}
  MinIO S3       ${PUBLIC}:{{ cookiecutter.nodeport_minio_api }}

Через Traefik (80): ${PUBLIC}/admin /chat /api /mlflow{% if cookiecutter.use_monitoring == "yes" %} /grafana{% endif %}

{% if cookiecutter.use_auth_service == "yes" %}
Логины:
  admin UI — admin + AUTH_BOOTSTRAP_PASSWORD из secrets.yaml
{% else %}
Inference API key — INFERENCE_API_KEY в secrets.yaml
{% endif %}
{% if cookiecutter.use_monitoring == "yes" %}
  Grafana  — admin + GF_SECURITY_ADMIN_PASSWORD
{% endif %}
  MinIO    — MINIO_ACCESS_KEY / MINIO_SECRET_KEY

На роутере пробрось: {{ cookiecutter.nodeport_admin }} {{ cookiecutter.nodeport_chat }}{% if cookiecutter.use_monitoring == "yes" %} {{ cookiecutter.nodeport_grafana }}{% endif %} {{ cookiecutter.nodeport_mlflow }} {{ cookiecutter.nodeport_app }} {{ cookiecutter.nodeport_minio_api }} {{ cookiecutter.nodeport_minio_console }} 80

EOF
