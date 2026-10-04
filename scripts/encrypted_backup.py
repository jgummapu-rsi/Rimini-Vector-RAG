from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger(__name__)
MAGIC = b"RAGBACKUP1"
BLOCK_SIZE = 1024 * 1024


def _key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("Backup key must contain exactly 32 random bytes")
    if path.stat().st_mode & 0o077:
        raise ValueError("Backup key must be accessible only to its owner")
    return key


def _read_exact(source, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        raise ValueError("Encrypted backup is truncated")
    return data


def _publish(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def encrypt(source: Path, destination: Path, key_file: Path) -> dict:
    cipher = AESGCM(_key(key_file))
    prefix = os.urandom(8)
    header = MAGIC + prefix
    digest = hashlib.sha256()
    total = 0
    counter = 0
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Backup destination must be a new file")
    temporary = None
    try:
        with (
            source.open("rb") as incoming,
            NamedTemporaryFile(dir=destination.parent, delete=False) as outgoing,
        ):
            temporary = Path(outgoing.name)
            outgoing.write(header)
            while block := incoming.read(BLOCK_SIZE):
                digest.update(block)
                total += len(block)
                nonce = prefix + struct.pack("!I", counter)
                ciphertext = cipher.encrypt(nonce, block, header + b"data")
                outgoing.write(struct.pack("!I", len(ciphertext)))
                outgoing.write(ciphertext)
                counter += 1
            manifest = json.dumps(
                {"bytes": total, "sha256": digest.hexdigest(), "blocks": counter}
            ).encode()
            ciphertext = cipher.encrypt(
                prefix + struct.pack("!I", counter), manifest, header + b"end"
            )
            outgoing.write(struct.pack("!I", 0))
            outgoing.write(struct.pack("!I", len(ciphertext)))
            outgoing.write(ciphertext)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _publish(temporary, destination)
        return {"bytes": total, "sha256": digest.hexdigest(), "blocks": counter}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def decrypt(source: Path, destination: Path, key_file: Path) -> dict:
    cipher = AESGCM(_key(key_file))
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Restore destination must be a new file")
    temporary = None
    digest = hashlib.sha256()
    total = counter = 0
    try:
        with (
            source.open("rb") as incoming,
            NamedTemporaryFile(dir=destination.parent, delete=False) as outgoing,
        ):
            temporary = Path(outgoing.name)
            header = _read_exact(incoming, len(MAGIC) + 8)
            if not header.startswith(MAGIC):
                raise ValueError("Unsupported encrypted backup format")
            prefix = header[len(MAGIC) :]
            while True:
                size = struct.unpack("!I", _read_exact(incoming, 4))[0]
                if size == 0:
                    size = struct.unpack("!I", _read_exact(incoming, 4))[0]
                    if not 16 <= size <= 1024:
                        raise ValueError("Invalid encrypted backup manifest")
                    manifest = json.loads(
                        cipher.decrypt(
                            prefix + struct.pack("!I", counter),
                            _read_exact(incoming, size),
                            header + b"end",
                        )
                    )
                    if incoming.read(1) or manifest != {
                        "bytes": total,
                        "sha256": digest.hexdigest(),
                        "blocks": counter,
                    }:
                        raise ValueError("Backup integrity manifest mismatch")
                    break
                if not 16 <= size <= BLOCK_SIZE + 16:
                    raise ValueError("Invalid encrypted backup block length")
                block = cipher.decrypt(
                    prefix + struct.pack("!I", counter),
                    _read_exact(incoming, size),
                    header + b"data",
                )
                digest.update(block)
                total += len(block)
                counter += 1
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _publish(temporary, destination)
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("encrypt", "decrypt"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    result = (encrypt if args.operation == "encrypt" else decrypt)(
        args.source, args.destination, args.key_file
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
