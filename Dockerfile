FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    APP_ROOT=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY run_app.sh ./run_app
RUN chmod +x run_app

ENV PATH="/app:${PATH}"

CMD ["run_app"]
