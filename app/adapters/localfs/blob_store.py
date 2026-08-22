"""Local filesystem BlobStore: content-addressed under data/blobs/<tenant>/."""
from __future__ import annotations

from pathlib import Path

from app.ports.blob_store import BlobStore


class LocalFsBlobStore(BlobStore):
    def __init__(self, root: Path):
        self.root = root

    def put(self, tenant_id: str, sha256: str, ext: str, data: bytes) -> str:
        tenant_dir = self.root / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        if ext and not ext.startswith("."):
            ext = "." + ext
        path = tenant_dir / f"{sha256}{ext}"
        if not path.exists():  # content-addressed: identical bytes = same file
            path.write_bytes(data)
        return str(path)

    def get(self, blob_path: str) -> bytes:
        return Path(blob_path).read_bytes()

    def delete(self, blob_path: str) -> None:
        p = Path(blob_path)
        if p.exists():
            p.unlink()
