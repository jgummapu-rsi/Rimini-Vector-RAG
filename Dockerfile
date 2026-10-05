# Single image, shared by the `api` and `worker` docker-compose services --
# same package, same requirements.txt; only the CMD/command differs.
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV's inference package requires these shared libraries on slim Linux.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime, psycopg2-binary, pandas/lxml/pillow all ship manylinux wheels for
# this base image -- no compiler / apt-get build-essential needed.
COPY requirements.txt ./
COPY vendor/docling/ ./vendor/docling/
RUN pip install --no-cache-dir -r requirements.txt && pip check

COPY app/ ./app/
COPY scripts/ ./scripts/

RUN groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --home-dir /app rag \
    && mkdir -p /app/data /app/models && chown -R rag:rag /app/data /app/models

# Named volumes are initialized from image contents. Persist the runtime user's
# ownership in the image so a newly created blob volume is writable immediately.
VOLUME ["/app/data", "/app/models"]

USER 10001:10001

EXPOSE 8000

# stdlib urllib instead of curl, so no extra apt-get layer is needed.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/readyz', timeout=5)" || exit 1

# Default (api service). The worker service overrides this via `command:`.
CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
