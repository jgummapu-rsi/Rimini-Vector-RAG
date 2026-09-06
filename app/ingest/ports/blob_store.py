"""BlobStore port: raw file bytes, content-addressed.

Local adapter = local filesystem. Production adapter = Azure Blob Storage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BlobStore(ABC):
    """Port for content-addressed raw file storage."""

    @abstractmethod
    def put(self, tenant_id: str, sha256: str, ext: str, data: bytes) -> str:
        """Store bytes, return an opaque blob path/URI."""

    @abstractmethod
    def get(self, blob_path: str) -> bytes:
        """Fetch the raw bytes previously stored at `blob_path`."""

    @abstractmethod
    def delete(self, blob_path: str) -> None:
        """Remove the blob at `blob_path`."""
