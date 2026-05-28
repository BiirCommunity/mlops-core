FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY auth_service ./auth_service

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    AUTH_DATABASE_PATH=/data/auth.db

VOLUME ["/data"]
EXPOSE 8090

CMD ["python", "-m", "auth_service.main"]
