"""BSON-compatible ObjectId generation (no pymongo dependency).

A 12-byte ObjectId = 4-byte epoch seconds + 5-byte per-process random +
3-byte incrementing counter, rendered as 24 hex chars. This is wire-compatible
with MongoDB ObjectIds, so tenant_id / user_id / document_id values will remain
valid if we later move metadata into Mongo/pymongo.
"""
from __future__ import annotations

import os
import re
import struct
import threading
import time

_lock = threading.Lock()

# Exactly 24 lowercase-or-uppercase hex digits, anchored. Deliberately a regex
# rather than int(value, 16): int() also accepts a leading sign, underscore
# digit separators and surrounding whitespace, so "+" + 23 hex chars and
# "ffff_ffff..." both passed the old check while being invalid ObjectIds.
_OBJECT_ID_RE = re.compile(r"\A[0-9a-fA-F]{24}\Z")
_machine = os.urandom(5)
_counter = int.from_bytes(os.urandom(3), "big")


def new_object_id() -> str:
    """Return a new 24-char hex ObjectId string."""
    global _counter
    ts = int(time.time())
    with _lock:
        _counter = (_counter + 1) % 0x1000000
        counter = _counter
    return (
        struct.pack(">I", ts).hex()
        + _machine.hex()
        + struct.pack(">I", counter)[1:].hex()  # low 3 bytes
    )


def is_object_id(value: str) -> bool:
    """True if `value` is exactly 24 hex characters."""
    return isinstance(value, str) and _OBJECT_ID_RE.match(value) is not None


def chunk_id(document_id: str, ordinal: int, total: int) -> str:
    """Deterministic chunk/point identity: document id + 1-based, zero-padded
    ordinal (e.g. <doc>001).

    Width scales with the chunk count so ids still sort lexicographically past
    999. Deriving it here -- rather than inside whichever metadata store happens
    to persist the chunk -- keeps the vector store and the metadata store
    agreeing on one identity without either depending on the other's side
    effects.
    """
    width = max(3, len(str(total)))
    return f"{document_id}{ordinal + 1:0{width}d}"
