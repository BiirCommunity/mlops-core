"""MinIO object storage for datasets and model artifacts."""

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from minio import Minio
from minio.error import S3Error

from app.training.config import TrainingSettings


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got: {uri}")
    without_scheme = uri[5:]
    bucket, _, object_name = without_scheme.partition("/")
    if not bucket or not object_name:
        raise ValueError(f"invalid s3 URI: {uri}")
    return bucket, object_name


class MinioStorage:
    def __init__(self, settings: TrainingSettings | None = None) -> None:
        self.settings = settings or TrainingSettings.from_env()
        self.client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
        )

    def ensure_buckets(self) -> None:
        for bucket in (
            self.settings.minio_bucket_models,
            self.settings.minio_bucket_datasets,
        ):
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        path = Path(file_path)
        self.client.fput_object(
            bucket,
            object_name,
            str(path),
            content_type=content_type,
        )
        return f"s3://{bucket}/{object_name}"

    def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        stream = io.BytesIO(payload)
        self.client.put_object(
            bucket,
            object_name,
            stream,
            length=len(payload),
            content_type=content_type,
        )
        return f"s3://{bucket}/{object_name}"

    def download_file(
        self, bucket: str, object_name: str, dest_path: str | Path
    ) -> Path:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(bucket, object_name, str(dest))
        return dest

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        response = self.client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        return [
            item.object_name for item in self.client.list_objects(bucket, prefix=prefix)
        ]

    def list_dataset_objects(self, prefix: str = "datasets/") -> list[dict[str, Any]]:
        bucket = self.settings.minio_bucket_datasets
        self.ensure_buckets()
        items: list[dict[str, Any]] = []
        for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True):
            name = obj.object_name
            if not name.endswith(".jsonl"):
                continue
            last_modified: datetime | None = obj.last_modified
            items.append(
                {
                    "object_name": name,
                    "filename": Path(name).name,
                    "uri": self.object_uri(bucket, name),
                    "size_bytes": obj.size or 0,
                    "last_modified": (
                        last_modified.isoformat() if last_modified else None
                    ),
                }
            )
        return sorted(items, key=lambda item: item["filename"].lower())

    def download_dataset_uri(
        self, uri: str, dest_dir: str | Path | None = None
    ) -> Path:
        bucket, object_name = parse_s3_uri(uri)
        target_dir = Path(dest_dir or self.settings.training_data_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / Path(object_name).name
        return self.download_file(bucket, object_name, local_path)

    def object_uri(self, bucket: str, object_name: str) -> str:
        return f"s3://{bucket}/{object_name}"

    def ping(self) -> tuple[bool, str]:
        try:
            self.ensure_buckets()
            return True, "ok"
        except S3Error as exc:
            return False, str(exc)
        except Exception as exc:  # pylint: disable=broad-except
            return False, str(exc)
