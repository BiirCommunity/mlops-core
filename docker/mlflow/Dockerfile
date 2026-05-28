FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow==2.22.0" boto3

WORKDIR /mlflow
VOLUME ["/mlflow"]

ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root s3://mlops-models/mlflow-artifacts/
