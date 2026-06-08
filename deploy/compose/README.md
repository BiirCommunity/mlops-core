# Локальный Docker Compose (legacy / dev)

Основной способ деплоя — **k3s**, см. [docs/deploy-argocd.md](../../docs/deploy-argocd.md).

```bash
cp ../../.env.docker.compose.example ../../.env.docker.compose
# заполните секреты

docker compose --env-file ../../.env.docker.compose -f docker-compose.yml up -d
```

Порты по умолчанию: Admin `:3000`, Chat `:3010`, API `:8080`, MLflow `:5000`, Grafana `:4345`.
