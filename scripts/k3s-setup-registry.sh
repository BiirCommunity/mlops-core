#!/usr/bin/env bash
# Локальный Docker Registry для k3s (сборка → push → pull).
set -euo pipefail

REGISTRY_NAME="${MLOPS_REGISTRY_NAME:-mlops-registry}"
REGISTRY_PORT="${MLOPS_REGISTRY_PORT:-5000}"

echo "==> Запуск registry на порту ${REGISTRY_PORT}..."
if docker ps -q -f "name=^/${REGISTRY_NAME}$" | grep -q .; then
  echo "Контейнер ${REGISTRY_NAME} уже запущен."
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

echo ""
echo "==> Настройка k3s (один раз, sudo)..."
if [[ -f "${REGISTRIES_DST}" ]] && grep -q 'localhost:5000' "${REGISTRIES_DST}" 2>/dev/null; then
  echo "registries.yaml уже содержит localhost:5000"
else
  echo "Скопируйте конфиг registry для k3s:"
  echo "  sudo cp ${REGISTRIES_SRC} ${REGISTRIES_DST}"
  echo "  sudo systemctl restart k3s"
fi

echo ""
echo "==> Docker: insecure-registries (один раз, sudo)..."
echo "Добавьте в /etc/docker/daemon.json фрагмент из deploy/k3s/daemon.json.snippet"
echo "Затем: sudo systemctl restart docker"
echo ""
echo "Проверка push: docker pull alpine && docker tag alpine localhost:${REGISTRY_PORT}/test:latest && docker push localhost:${REGISTRY_PORT}/test:latest"
