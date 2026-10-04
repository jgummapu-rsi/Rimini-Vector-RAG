import hashlib
import os
from pathlib import Path

import pytest

from app.ingest.adapters.localfs.blob_store import LocalFsBlobStore


def test_put_get_roundtrip(tmp_path):
    blob = LocalFsBlobStore(tmp_path)
    path = blob.put("T1", hashlib.sha256(b"hello").hexdigest(), ".txt", b"hello")
    assert blob.get(path) == b"hello"
    assert blob.size(path) == 5
    assert blob.get_range(path, 1, 3) == b"ell"


def test_invalid_byte_range_is_rejected(tmp_path):
    blob = LocalFsBlobStore(tmp_path)
    path = blob.put("T1", hashlib.sha256(b"hello").hexdigest(), ".txt", b"hello")
    with pytest.raises(ValueError, match="range"):
        blob.get_range(path, 4, 5)


def test_content_addressed_and_tenant_scoped(container):
    sha = hashlib.sha256(b"data").hexdigest()
    p1 = container.blob.put("T1", sha, ".pdf", b"data")
    p2 = container.blob.put("T1", sha, ".pdf", b"data")
    assert p1 == p2
    p3 = container.blob.put("T2", sha, ".pdf", b"data")
    assert p3 != p1


def test_delete(container):
    path = container.blob.put("T1", hashlib.sha256(b"x").hexdigest(), ".bin", b"x")
    container.blob.delete(path)
    with pytest.raises(FileNotFoundError):
        container.blob.get(path)
    container.blob.delete(path)


def test_partial_blob_is_repaired_on_retry(container):
    data = b"complete source evidence"
    sha = hashlib.sha256(data).hexdigest()
    path = Path(container.blob.put("T1", sha, ".txt", data))
    path.write_bytes(data[:5])
    with pytest.raises(ValueError, match="hash verification"):
        container.blob.get(str(path))
    assert container.blob.put("T1", sha, ".txt", data) == str(path)
    assert container.blob.get(str(path)) == data


def test_atomic_write_failure_preserves_original_blob(container, monkeypatch):
    data = b"source evidence"
    sha = hashlib.sha256(data).hexdigest()

    def fail(*args):
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="rename failed"):
        container.blob.put("T1", sha, ".txt", data)
    assert list((container.settings.blob_dir / "T1").iterdir()) == []
