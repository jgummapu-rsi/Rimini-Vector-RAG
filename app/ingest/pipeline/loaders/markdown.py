"""Markdown loader (.md): table-aware via the shared block splitter."""
from __future__ import annotations

from app.ingest.pipeline.blocks import split_blocks


def extract(data: bytes, filename: str, gateway, cfg) -> list:
    """Parse a markdown file into prose/table Elements (cfg unused: no images, no tables to cap)."""
    text = data.decode("utf-8", "ignore")
    return split_blocks(
        text,
        text_extractor="markdown",
        table_extractor="markdown_table",
        text_reason="prose",
        table_reason="structured_table",
    )
