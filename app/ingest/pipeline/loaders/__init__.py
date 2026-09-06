"""Loader dispatch: raw bytes -> ordered Elements, plus a route summary.

Routing is inherently per-loader (PDF routes differ from Excel), so each loader
owns its own parse+route+extract. This module picks the loader by extension and
computes a uniform route summary from the produced elements.
"""
from __future__ import annotations

import os
from collections import Counter

from app.shared.config import Settings
from app.shared.gateway.client import LiteLLMClient
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.loaders import docx_loader, excel, image, markdown, pdf, table, text

_DISPATCH = {
    ".pdf": pdf.extract,
    ".docx": docx_loader.extract,
    ".md": markdown.extract,
    ".txt": text.extract,
    ".rtf": text.extract,
    ".png": image.extract,
    ".jpg": image.extract,
    ".jpeg": image.extract,
    ".tiff": image.extract,
    ".tif": image.extract,
    ".webp": image.extract,
    ".csv": table.extract_csv,
    ".html": table.extract_html,
    ".htm": table.extract_html,
    ".xlsx": excel.extract,
    ".xls": excel.extract,
}


def extract_document(
    filename: str, data: bytes, gateway: LiteLLMClient, cfg: Settings
) -> tuple[list[Element], dict]:
    """Dispatch to the per-extension loader, passing file-safety limits through."""
    ext = os.path.splitext(filename or "")[1].lower()
    loader = _DISPATCH.get(ext)
    if loader is None:
        raise ValueError(f"no loader for extension '{ext}'")
    elements = loader(data, filename, gateway, cfg)
    return elements, _route_summary(elements)


def _route_summary(elements: list[Element]) -> dict:
    """Aggregate per-element extractor/modality/route_reason counts for the job trace."""
    by_extractor = Counter(e.extractor for e in elements)
    by_modality = Counter(e.modality for e in elements)
    reasons = Counter(e.route_reason for e in elements)
    return {
        "elements": len(elements),
        "by_extractor": dict(by_extractor),
        "by_modality": dict(by_modality),
        "by_reason": dict(reasons),
    }
