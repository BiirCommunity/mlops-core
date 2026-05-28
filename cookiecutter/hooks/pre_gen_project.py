#!/usr/bin/env python3

import re
import sys


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


slug = "{{ cookiecutter.project_slug }}"
namespace = "{{ cookiecutter.namespace }}"
prefix = "{{ cookiecutter.auth_token_prefix }}"

if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
    fail(f"project_slug must be lowercase alphanumeric with hyphens: {slug!r}")

if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", namespace):
    fail(f"namespace must be lowercase alphanumeric with hyphens: {namespace!r}")

if not prefix or not re.fullmatch(r"[a-zA-Z0-9_]+$", prefix):
    fail("auth_token_prefix must be a non-empty alphanumeric/underscore prefix")
