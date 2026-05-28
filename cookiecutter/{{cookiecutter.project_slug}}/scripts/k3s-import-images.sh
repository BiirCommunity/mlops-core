#!/usr/bin/env bash
echo "Устарело: образы больше не импортируются через docker save."
echo "Используйте сборку и push в локальный registry:"
echo "  ./scripts/k3s-setup-registry.sh   # один раз"
echo "  ./scripts/k3s-build-images.sh"
exit 1
