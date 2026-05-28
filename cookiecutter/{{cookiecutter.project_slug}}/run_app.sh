#!/bin/sh
set -eu
cd "${APP_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

case "${RUN_MODE:-}" in
  WEBSERVER)
    exec uv run uvicorn app.app:app \
      --host "${FASTAPI_HOST:-localhost}" \
      --port "${FASTAPI_PORT:-8080}"
    ;;
  *)
    echo "RUN_MODE must be WEBSERVER (got: ${RUN_MODE:-<unset>})" >&2
    exit 1
    ;;
esac
