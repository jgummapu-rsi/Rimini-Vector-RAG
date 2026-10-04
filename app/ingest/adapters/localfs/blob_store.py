"""Local filesystem BlobStore: content-addressed under data/blobs/<tenant>/."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

from app.ingest.ports.blob_store import BlobStore

log = logging.getLogger(__name__)


class LocalFsBlobStore(BlobStore):
    def __init__(self, root: Path):
        self.root = root

    def put(self, tenant_id: str, sha256: str, ext: str, data: bytes) -> str:
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Blob content does not match its SHA-256 identity")
        if not tenant_id or Path(tenant_id).name != tenant_id or tenant_id in {".", ".."}:
            raise ValueError("Invalid blob tenant path")
        tenant_dir = self.root / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        if ext and not ext.startswith("."):
            ext = "." + ext
        if "/" in ext or "\\" in ext:
            raise ValueError("Invalid blob extension")
        path = tenant_dir / f"{sha256}{ext}"
        if path.exists():
            existing = path.read_bytes()
            if len(existing) == len(data) and hashlib.sha256(existing).hexdigest() == sha256:
                return str(path)
            log.warning(
                "Incomplete content-addressed blob replaced",
                extra={"event": "blob_repaired", "tenant_id": tenant_id, "content_sha256": sha256},
            )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=tenant_dir, prefix=".upload-", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(tenant_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return str(path)

    def get(self, blob_path: str) -> bytes:
        path = Path(blob_path)
        data = path.read_bytes()
        expected = path.name.split(".", 1)[0]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Stored blob failed its content hash verification")
        return data

    def size(self, blob_path: str) -> int:
        return Path(blob_path).stat().st_size

    def get_range(self, blob_path: str, start: int, end: int) -> bytes:
        size = self.size(blob_path)
        if start < 0 or end < start or end >= size:
            raise ValueError("Invalid blob byte range")
        with Path(blob_path).open("rb") as source:
            source.seek(start)
            return source.read(end - start + 1)

    def delete(self, blob_path: str) -> None:
        p = Path(blob_path)
        if p.exists():
            p.unlink()
