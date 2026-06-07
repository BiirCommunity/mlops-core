#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${ROOT}/models/model.pt"
NS={{ cookiecutter.namespace }}
HELPER=model-copy-helper

if [[ ! -f "$MODEL" ]]; then
  echo "Нет файла: $MODEL"
  echo "dvc pull или положи checkpoint в models/model.pt"
  exit 1
fi

cleanup() {
  kubectl -n "$NS" delete pod "$HELPER" --ignore-not-found --wait=false 2>/dev/null || true
}
trap cleanup EXIT

kubectl -n "$NS" delete pod "$HELPER" --ignore-not-found --wait=true 2>/dev/null || true

kubectl -n "$NS" run "$HELPER" \
  --restart=Never \
  --image=busybox:1.36 \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "helper",
        "image": "busybox:1.36",
        "command": ["sleep", "3600"],
        "volumeMounts": [{"name": "models", "mountPath": "/models"}]
      }],
      "volumes": [{
        "name": "models",
        "persistentVolumeClaim": {"claimName": "models-pvc"}
      }]
    }
  }'

kubectl -n "$NS" wait --for=condition=Ready "pod/${HELPER}" --timeout=120s

echo "Копирую model.pt ($(du -h "$MODEL" | cut -f1))..."
kubectl -n "$NS" cp "$MODEL" "${HELPER}:/models/model.pt"
kubectl -n "$NS" exec "$HELPER" -- ls -lh /models/model.pt

kubectl -n "$NS" rollout restart deployment/app
kubectl -n "$NS" rollout status deployment/app --timeout=900s

echo "Готово. Health: http://localhost:{{ cookiecutter.nodeport_app }}/health"
