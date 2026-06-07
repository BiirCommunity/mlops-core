FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    APP_ROOT=/app

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY app ./app
COPY run_app.sh ./run_app
RUN chmod +x run_app

ENV PORT=8080 \
    RUN_MODE=WEBSERVER \
    FASTAPI_HOST=0.0.0.0 \
    FASTAPI_PORT=8080

EXPOSE 8080

CMD ["run_app"]
