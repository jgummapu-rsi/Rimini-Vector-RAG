import os

import pytest
from cryptography.exceptions import InvalidTag

from scripts.encrypted_backup import BLOCK_SIZE, decrypt, encrypt


def test_encrypted_backup_roundtrip_and_independent_destination(tmp_path):
    key = tmp_path / "key"
    key.write_bytes(os.urandom(32))
    key.chmod(0o600)
    source = tmp_path / "dump"
    data = os.urandom(BLOCK_SIZE + 157)
    source.write_bytes(data)
    archive = tmp_path / "archive.enc"
    expected = encrypt(source, archive, key)
    source.unlink()
    assert decrypt(archive, source, key) == expected
    assert source.read_bytes() == data


@pytest.mark.parametrize("damage", ["truncate", "tamper", "append", "wrong_key"])
def test_damaged_backup_never_publishes_partial_plaintext(tmp_path, damage):
    key = tmp_path / "key"
    key.write_bytes(os.urandom(32))
    key.chmod(0o600)
    source = tmp_path / "source"
    source.write_bytes(b"source evidence" * 100)
    archive = tmp_path / "backup"
    encrypt(source, archive, key)
    data = archive.read_bytes()
    if damage == "truncate":
        archive.write_bytes(data[:-20])
    elif damage == "append":
        archive.write_bytes(data + b"extra")
    elif damage == "tamper":
        archive.write_bytes(data[:30] + bytes([data[30] ^ 1]) + data[31:])
    else:
        key.write_bytes(os.urandom(32))
    restored = tmp_path / "restore"
    with pytest.raises((InvalidTag, ValueError)):
        decrypt(archive, restored, key)
    assert not restored.exists()


def test_backup_refuses_insecure_key_and_overwrite(tmp_path):
    key = tmp_path / "key"
    key.write_bytes(os.urandom(32))
    key.chmod(0o644)
    source = tmp_path / "source"
    source.write_bytes(b"data")
    with pytest.raises(ValueError, match="owner"):
        encrypt(source, tmp_path / "encrypted", key)
    key.chmod(0o600)
    with pytest.raises(ValueError, match="new file"):
        encrypt(source, source, key)
