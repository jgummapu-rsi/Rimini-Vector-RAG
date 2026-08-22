"""Standalone image loader: strict LLM transcription, table-aware.

If the image (or a region of it) is a table, it becomes a TABLE element via the
shared block splitter; otherwise a description/transcription as an IMAGE element.
"""
from __future__ import annotations

import os

from app.domain.models import Modality
from app.pipeline.blocks import split_blocks
from app.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT

_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".tiff": "image/tiff", ".tif": "image/tiff", ".webp": "image/webp",
}


def extract(data: bytes, filename: str, gateway) -> list:
    ext = os.path.splitext(filename or "")[1].lower()
    mime = _MIME.get(ext, "image/png")

    out = gateway.vision(data, STRICT_TRANSCRIBE_PROMPT, mime).strip()
    if not out:
        return []
    return split_blocks(
        out,
        text_extractor="vision",
        table_extractor="vision_table",
        text_reason="image_asset",
        table_reason="image_table",
        text_modality=Modality.IMAGE.value,
    )
