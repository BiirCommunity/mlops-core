FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.135.1" "uvicorn>=0.42.0"

COPY auth_service ./auth_service

EXPOSE 8090

CMD ["uvicorn", "auth_service.main:app", "--host", "0.0.0.0", "--port", "8090"]
