# Деплой в k3s

## Предварительные условия

k3s, `kubectl`, локальный registry (`./scripts/k3s-setup-registry.sh`).

GPU опционален — overlay `k8s/overlays/no-gpu` или `include_gpu=no` при генерации.

## Шаги

```bash
cp k8s/secrets.example.yaml k8s/secrets.yaml
./scripts/k3s-build-images.sh
./scripts/k3s-deploy.sh
```

Секреты: `INFERENCE_API_KEY`, `MINIO_*`, при auth — `AUTH_BOOTSTRAP_PASSWORD`, при monitoring — `GF_SECURITY_ADMIN_*`.

NodePort'ы задаются при генерации проекта (`cookiecutter.json`).
