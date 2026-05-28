import subprocess
from pathlib import Path


def test_model_dvc_metadata_exists() -> None:
    dvc_file = Path("models/model.pt.dvc")
    assert dvc_file.is_file()
    text = dvc_file.read_text(encoding="utf-8")
    assert "path: model.pt" in text
    assert "md5:" in text
    assert dvc_file.stat().st_size > 0


def test_dvc_remote_configured() -> None:
    config = Path(".dvc/config").read_text(encoding="utf-8")
    assert 'remote "minio"' in config
    assert "s3://{{ cookiecutter.minio_bucket_models }}/dvc" in config
    assert "endpointurl" in config


def test_dvc_setup_script_exists() -> None:
    script = Path("scripts/dvc-setup.sh")
    assert script.is_file()
    assert script.stat().st_mode & 0o111
