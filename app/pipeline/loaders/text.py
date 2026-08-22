"""Plain text / markdown / rtf loader (no external calls).

Runs through the shared block splitter so any GFM tables inside a .txt/.rtf dump
are treated as tables (consistent with markdown files), not buried in prose.
"""
from __future__ import annotations

import os

from app.pipeline.blocks import split_blocks


def extract(data: bytes, filename: str, gateway) -> list:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".rtf":
        from striprtf.striprtf import rtf_to_text

        content = rtf_to_text(data.decode("utf-8", "ignore"))
    else:
        content = data.decode("utf-8", "ignore")

    content = content.strip()
    if not content:
        return []
    return split_blocks(
        content,
        text_extractor="text",
        table_extractor="text_table",
        text_reason="text_layer",
        table_reason="structured_table",
    )
