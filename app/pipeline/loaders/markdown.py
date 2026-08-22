"""Markdown loader (.md): table-aware via the shared block splitter."""
from __future__ import annotations

from app.pipeline.blocks import split_blocks


def extract(data: bytes, filename: str, gateway) -> list:
    text = data.decode("utf-8", "ignore")
    return split_blocks(
        text,
        text_extractor="markdown",
        table_extractor="markdown_table",
        text_reason="prose",
        table_reason="structured_table",
    )
