#!/usr/bin/env bash

REGISTRY_NAME="${MLOPS_REGISTRY_NAME:-mlops-registry}"
REGISTRY_PORT="${MLOPS_REGISTRY_PORT:-5000}"

echo "Registry на порту ${REGISTRY_PORT}..."
if docker ps -q -f "name=^/${REGISTRY_NAME}$" | grep -q .; then
  echo "${REGISTRY_NAME} уже работает"
elif docker ps -aq -f "name=^/${REGISTRY_NAME}$" | grep -q .; then
  docker start "${REGISTRY_NAME}"
else
  docker run -d \
    --restart=unless-stopped \
    -p "127.0.0.1:${REGISTRY_PORT}:5000" \
    --name "${REGISTRY_NAME}" \
    registry:2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRIES_SRC="${ROOT}/deploy/k3s/registries.yaml"
REGISTRIES_DST="/etc/rancher/k3s/registries.yaml"

echo
if [[ -f "${REGISTRIES_DST}" ]] && grep -q '{{ cookiecutter.mlops_registry }}' "${REGISTRIES_DST}" 2>/dev/null; then
  echo "k3s уже настроен ({{ cookiecutter.mlops_registry }} в registries.yaml)"
else
  echo "Один раз с sudo:"
  echo "  cp ${REGISTRIES_SRC} ${REGISTRIES_DST}"
  echo "  systemctl restart k3s"
fi

echo
echo "Docker — добавить insecure-registries из deploy/k3s/daemon.json.snippet,"
echo "потом: sudo systemctl restart docker"
echo
echo "Проверка:"
echo "  docker pull alpine"
echo "  docker tag alpine localhost:${REGISTRY_PORT}/test:latest"
echo "  docker push localhost:${REGISTRY_PORT}/test:latest"
