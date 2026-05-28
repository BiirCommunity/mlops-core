# Локальный registry для k3s

Образы **собираются** (`docker build`) и **пушатся** в registry на `localhost:5000`. k3s **pull** при старте подов. Перенос через `docker save | ctr import` не используется.

## Один раз

### 1. Registry

```bash
./scripts/k3s-setup-registry.sh
```

### 2. k3s

```bash
sudo cp deploy/k3s/registries.yaml /etc/rancher/k3s/registries.yaml
sudo systemctl restart k3s
```

### 3. Docker (insecure registry)

Добавьте в `/etc/docker/daemon.json` содержимое `deploy/k3s/daemon.json.snippet` (слить с существующим JSON), затем:

```bash
sudo systemctl restart docker
```

## Каждый деплой / после изменения кода

```bash
./scripts/k3s-build-images.sh   # build + push
./scripts/k3s-deploy.sh          # если ещё не применяли манифесты
./scripts/k3s-copy-model.sh    # model.pt в PVC
```

Другой registry: `MLOPS_REGISTRY=192.168.0.103:5000 ./scripts/k3s-build-images.sh` и обновите `images:` в `k8s/kustomization.yaml`.
