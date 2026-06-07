#!/usr/bin/env python3

import re
import shutil
from pathlib import Path

PROJECT = Path(".").resolve()


def rm(path: str) -> None:
    target = PROJECT / path
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif target.is_file():
        target.unlink(missing_ok=True)


{% if cookiecutter.use_auth_service == "no" %}
rm("auth-service")
rm("tests/test_auth_service.py")
{% endif %}

{% if cookiecutter.use_monitoring == "no" %}
rm("prometheus")
rm("grafana-provisioning")
rm("k8s/base/grafana")
{% endif %}

compose = PROJECT / "deploy/compose/docker-compose.yml"
if compose.is_file():
    text = compose.read_text(encoding="utf-8")
{% if cookiecutter.use_auth_service == "no" %}
    text = re.sub(
        r"\n  auth-service:.*?(?=\n  [a-zA-Z0-9_-]+:|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
{% endif %}
{% if cookiecutter.use_monitoring == "no" %}
    for service in ("prometheus", "grafana"):
        text = re.sub(
            rf"\n  {service}:.*?(?=\n  [a-zA-Z0-9_-]+:|\Z)",
            "",
            text,
            flags=re.DOTALL,
        )
{% endif %}
{% if cookiecutter.include_gpu == "no" %}
    text = re.sub(
        r"\n    deploy:\n      resources:\n        reservations:\n          devices:\n            - capabilities: \[gpu\]\n",
        "\n",
        text,
    )
{% endif %}
    compose.write_text(text, encoding="utf-8")
