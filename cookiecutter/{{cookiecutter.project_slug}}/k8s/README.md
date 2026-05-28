# k3s manifests

```bash
cp secrets.example.yaml secrets.yaml
./scripts/k3s-setup-registry.sh
./scripts/k3s-build-images.sh
kubectl apply -f secrets.yaml
kubectl apply -k .
./scripts/k3s-copy-model.sh
```

- `base/` — все ресурсы + NodePort (30000–30901)
- `overlays/no-gpu/` — CPU без NVIDIA GPU

Документация: [../docs/deploy-k3s.md](../docs/deploy-k3s.md) · [../deploy/k3s/README.md](../deploy/k3s/README.md)
