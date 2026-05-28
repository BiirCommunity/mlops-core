#!/usr/bin/env bash
# Configure DVC remote from .env.docker.compose (MinIO S3-compatible).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.docker.compose}"
DVC="${DVC_BIN:-$ROOT/.venv/bin/dvc}"

if [[ ! -x "$DVC" ]]; then
  echo "DVC not found. Run: uv sync --group dev" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

endpoint="${DVC_MINIO_ENDPOINT:-http://127.0.0.1:9000}"
if [[ "$endpoint" == minio:* ]]; then
  port="${endpoint#*:}"
  endpoint="http://127.0.0.1:${port}"
fi
if [[ "${MINIO_SECURE:-0}" == "1" ]]; then
  endpoint="${endpoint/http:/https:}"
fi

bucket="${DVC_BUCKET:-${MINIO_BUCKET_MODELS:-mlops-models}}"
prefix="${DVC_PREFIX:-dvc}"
access_key="${MINIO_ACCESS_KEY:-minioadmin}"
secret_key="${MINIO_SECRET_KEY:-minioadmin}"

"$DVC" remote modify minio url "s3://${bucket}/${prefix}"
"$DVC" remote modify minio endpointurl "$endpoint"
"$DVC" remote modify minio access_key_id "$access_key"
"$DVC" remote modify --local minio secret_access_key "$secret_key"

echo "DVC remote 'minio' -> s3://${bucket}/${prefix}"
echo "endpoint: ${endpoint}"
