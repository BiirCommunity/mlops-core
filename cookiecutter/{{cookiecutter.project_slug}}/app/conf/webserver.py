import os

FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", "8080"))
FASTAPI_HOST = os.environ.get("FASTAPI_HOST", "localhost")
