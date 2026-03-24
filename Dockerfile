# GPU-ориентированный образ (CUDA 12.1)
FROM ppytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

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

ENV PATH="/app:${PATH}" \
    DEVICE=cuda \
    CHECKPOINT_PATH=/models/model.pt \
    TOKENIZER_NAME=meta-llama/Meta-Llama-3-8B \
    MAX_CONTEXT_TOKENS=4096 \
    PORT=8080

EXPOSE 8080

CMD ["run_app"]

