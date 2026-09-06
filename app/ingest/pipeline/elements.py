"""`Element` = one unit of extracted content ready for chunking.

A loader turns a raw file into an ordered list of Elements. Each Element already
knows its modality (text/image/table), which extractor produced it, and why it
was routed there — that provenance flows onto every chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Element:
    text: str
    modality: str        # Modality value: text | image | table
    extractor: str       # pdf_text | pdf_table | ocr | vision | docx | csv_table | ...
    route_reason: str    # why the router chose this path
    order: int
    meta: dict[str, Any] = field(default_factory=dict)
