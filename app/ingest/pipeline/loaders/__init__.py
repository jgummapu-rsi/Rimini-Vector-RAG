"""Loader dispatch: raw bytes -> ordered Elements, plus a route summary.

Routing is inherently per-loader (PDF routes differ from Excel), so each loader
owns its own parse+route+extract. This module picks the loader by extension and
computes a uniform route summary from the produced elements.
"""

from __future__ import annotations

import os
from collections import Counter

from app.ingest.pipeline.docling_config import uses_docling
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.loaders import (
    docling_loader,
    docx_loader,
    excel,
    image,
    markdown,
    pdf,
    table,
    text,
)
from app.ingest.pipeline.safety import UnsafeContentError
from app.shared.config import Settings
from app.shared.gateway.client import LiteLLMClient

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
    if uses_docling(filename, cfg):
        loader = docling_loader.extract
    if loader is None:
        raise ValueError(f"no loader for extension '{ext}'")
    elements = loader(data, filename, gateway, cfg)
    for index, element in enumerate(elements):
        element.meta.setdefault("source_element", index)
        element.meta.setdefault(
            "source_spans",
            [
                {
                    "element": index,
                    "start": 0,
                    "end": len(element.text),
                    "text": element.text,
                    **{
                        key: element.meta[key]
                        for key in (
                            "page",
                            "bbox",
                            "page_width",
                            "page_height",
                            "frame",
                            "sheet",
                            "cell_range",
                            "row",
                            "column",
                            "row_start",
                            "row_end",
                            "source_line",
                            "source_line_start",
                            "source_line_end",
                            "body_index",
                            "paragraph_index",
                            "table_index",
                            "relationship_id",
                        )
                        if key in element.meta
                    },
                }
            ],
        )
        for span in element.meta["source_spans"]:
            span.setdefault("element", index)
            for key in ("page_box", "layout_id", "layout_label", "layout_confidence"):
                if key in element.meta:
                    span.setdefault(key, element.meta[key])
            span.setdefault(
                "precision",
                element.meta.get("precision")
                or (
                    "element_region"
                    if element.modality != "text" or element.extractor == "vision"
                    else "exact_text"
                ),
            )
    if (
        not any(element.text.strip() for element in elements)
        and data.strip()
        and ext not in {".pdf", ".txt", ".md", ".rtf"}
    ):
        raise UnsafeContentError("Nonempty source produced no extracted evidence")
    summary = _route_summary(elements)
    summary["extraction_status"] = (
        "partial"
        if summary["omitted_regions"]
        else "complete"
        if any(element.text.strip() for element in elements)
        else "empty"
    )
    if summary["omitted_regions"] and not any(element.text.strip() for element in elements):
        raise UnsafeContentError("Source has omitted regions and no extracted evidence")
    return elements, summary


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
        "omitted_regions": sum(bool(e.meta.get("omission")) for e in elements),
    }
