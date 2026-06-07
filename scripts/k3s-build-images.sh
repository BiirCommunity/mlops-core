#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REGISTRY="${MLOPS_REGISTRY:-localhost:5000}"
TAG="${MLOPS_IMAGE_TAG:-latest}"
PUBLIC_URL="${MLOPS_PUBLIC_URL:-http://83.221.210.29}"

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

build_push mlops-core-admin-ui ./admin-ui \
  --build-arg "VITE_MLFLOW_UI_URL=${PUBLIC_URL}:30500"
build_push mlops-core-chat-ui ./chat-ui
build_push mlops-core-mlflow ./docker/mlflow
build_push mlops-core-auth-service ./auth-service
build_push mlops-core-app .

if kubectl cluster-info >/dev/null 2>&1; then
  echo
  echo "kubectl apply + restart deployments..."
  kubectl apply -k k8s/ 2>/dev/null || true
  kubectl -n mlops rollout restart \
    deployment/app deployment/auth-service deployment/admin-ui \
    deployment/chat-ui deployment/mlflow 2>/dev/null || true
  echo "Смотри: kubectl -n mlops get pods -w"
else
  echo
  echo "kubectl недоступен — потом запусти ./scripts/k3s-deploy.sh"
fi

echo
echo "Образы в ${REGISTRY}"
