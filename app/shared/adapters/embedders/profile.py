from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def tokenizer_identity(repo: str, revision: str | None = None) -> tuple[str, str]:
    path = Path(hf_hub_download(repo, "tokenizer.json", revision=revision))
    resolved = path.parent.name
    if len(resolved) != 40 or any(char not in "0123456789abcdef" for char in resolved):
        raise ValueError("Tokenizer must resolve to an immutable model commit")
    return resolved, hashlib.sha256(path.read_bytes()).hexdigest()
