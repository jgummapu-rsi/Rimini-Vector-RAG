import pytest


def test_put_get_roundtrip(container):
    path = container.blob.put("T1", "sha1", ".txt", b"hello")
    assert container.blob.get(path) == b"hello"


def test_content_addressed_and_tenant_scoped(container):
    p1 = container.blob.put("T1", "shaX", ".pdf", b"data")
    p2 = container.blob.put("T1", "shaX", ".pdf", b"data")
    assert p1 == p2                        # same content -> same path
    p3 = container.blob.put("T2", "shaX", ".pdf", b"data")
    assert p3 != p1                        # different tenant dir


def test_delete(container):
    path = container.blob.put("T1", "sha9", ".bin", b"x")
    container.blob.delete(path)
    with pytest.raises(FileNotFoundError):
        container.blob.get(path)
    container.blob.delete(path)            # idempotent, no raise
