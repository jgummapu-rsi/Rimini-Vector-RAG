"""API bearer-token hashing: tokens are shown to the caller once, at mint time,
and only ever stored/compared as a one-way hash from then on."""
from __future__ import annotations

import hashlib


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a bearer token, for at-rest storage and lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
