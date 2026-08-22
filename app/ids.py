"""BSON-compatible ObjectId generation (no pymongo dependency).

A 12-byte ObjectId = 4-byte epoch seconds + 5-byte per-process random +
3-byte incrementing counter, rendered as 24 hex chars. This is wire-compatible
with MongoDB ObjectIds, so tenant_id / user_id / document_id values will remain
valid if we later move metadata into Mongo/pymongo.
"""
from __future__ import annotations

import os
import struct
import threading
import time

_lock = threading.Lock()
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
    """True if `value` looks like a 24-char hex ObjectId."""
    if not isinstance(value, str) or len(value) != 24:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False
