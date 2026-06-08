#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REGISTRY="${MLOPS_REGISTRY:-{{ cookiecutter.mlops_registry }}}"
TAG="${MLOPS_IMAGE_TAG:-{{ cookiecutter.mlops_image_tag }}}"
PUBLIC_URL="${MLOPS_PUBLIC_URL:-{{ cookiecutter.mlops_public_url }}}"

"${ROOT}/scripts/k3s-setup-registry.sh"

build_push() {
  local name="$1"
  local context="$2"
  shift 2
  local image="${REGISTRY}/${name}:${TAG}"

  echo
  echo "build ${image} (${context})"
  docker build -t "${image}" "$@" "${context}"
  docker push "${image}"
}

echo "Сборка → ${REGISTRY}, tag ${TAG}"

build_push {{ cookiecutter.project_slug }}-admin-ui ./admin-ui \
  --build-arg "VITE_MLFLOW_UI_URL=${PUBLIC_URL}:{{ cookiecutter.nodeport_mlflow }}"
build_push {{ cookiecutter.project_slug }}-chat-ui ./chat-ui
build_push {{ cookiecutter.project_slug }}-mlflow ./docker/mlflow
{% if cookiecutter.use_auth_service == "yes" %}
build_push {{ cookiecutter.project_slug }}-auth-service ./auth-service
{% endif %}
build_push {{ cookiecutter.project_slug }}-app .

if kubectl cluster-info >/dev/null 2>&1; then
  echo
  echo "kubectl apply + restart deployments..."
  kubectl apply -k k8s/ 2>/dev/null || true
  kubectl -n {{ cookiecutter.namespace }} rollout restart \
    deployment/app deployment/admin-ui \
    deployment/chat-ui deployment/mlflow{% if cookiecutter.use_auth_service == "yes" %} \
    deployment/auth-service{% endif %} 2>/dev/null || true
  echo "Смотри: kubectl -n {{ cookiecutter.namespace }} get pods -w"
else
  echo
  echo "kubectl недоступен — потом запусти ./scripts/k3s-deploy.sh"
fi

echo
echo "Образы в ${REGISTRY}"
