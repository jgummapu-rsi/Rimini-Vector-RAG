# Single image, shared by the `api` and `worker` docker-compose services --
# same package, same requirements.txt; only the CMD/command differs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# onnxruntime, psycopg2-binary, pandas/lxml/pillow all ship manylinux wheels for
# this base image -- no compiler / apt-get build-essential needed.
COPY requirements.txt ./
# Runtime image only -- requirements-dev.txt (pytest/jupyter/ragas) is dev/test
# -only and is never installed here.
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

EXPOSE 8000

# stdlib urllib instead of curl, so no extra apt-get layer is needed.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1

# Default (api service). The worker service overrides this via `command:`.
CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
