"""Convert private chunk metadata into a bounded, stable citation contract."""

from __future__ import annotations

import math
from typing import Any

_LOCATOR_KEYS = (
    "page",
    "pages",
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
    "section",
    "section_path",
    "relationship_id",
)
_MAX_REGIONS = 4096


def citation_provenance(meta: dict[str, Any] | None, source_type: str | None) -> dict[str, Any]:
    """Return only display-safe source locators; never expose arbitrary JSONB metadata."""
    meta = meta or {}
    locator = {key: meta[key] for key in _LOCATOR_KEYS if meta.get(key) is not None}
    regions: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    candidates = meta.get("source_spans") or [meta]
    for span in candidates:
        if not isinstance(span, dict) or not span.get("bbox"):
            continue
        try:
            x0, top, x1, bottom = (float(value) for value in span["bbox"][:4])
            width = float(span.get("page_width") or meta.get("page_width"))
            height = float(span.get("page_height") or meta.get("page_height"))
            page = int(span.get("page") or meta.get("page"))
        except (TypeError, ValueError, KeyError):
            continue
        if (
            not all(math.isfinite(v) for v in (x0, top, x1, bottom, width, height))
            or width <= 0
            or height <= 0
        ):
            continue
        page_box = span.get("page_box") or meta.get("page_box") or (0, 0, width, height)
        try:
            origin_x, origin_y, end_x, end_y = map(float, page_box)
        except (TypeError, ValueError):
            continue
        width, height = end_x - origin_x, end_y - origin_y
        if (
            not all(math.isfinite(v) for v in (origin_x, origin_y, end_x, end_y))
            or width <= 0
            or height <= 0
        ):
            continue
        left, right = max(0.0, x0 - origin_x), min(width, x1 - origin_x)
        upper, lower = max(0.0, top - origin_y), min(height, bottom - origin_y)
        if right <= left or lower <= upper:
            continue
        region = (
            page,
            round(left / width, 6),
            round(upper / height, 6),
            round((right - left) / width, 6),
            round((lower - upper) / height, 6),
        )
        if region in seen:
            continue
        seen.add(region)
        item = {
            "page": region[0],
            "x": region[1],
            "y": region[2],
            "width": region[3],
            "height": region[4],
            "source_element": span.get("element", meta.get("source_element")),
            "precision": span.get("precision", "exact_text"),
        }
        if span.get("text"):
            item["text"] = str(span["text"])[:1000]
        for key in ("layout_id", "layout_label", "layout_confidence"):
            if key in span:
                item[key] = span[key]
        regions.append(item)

    total = len(regions)
    regions = regions[:_MAX_REGIONS]
    pages = sorted({int(page) for page in (meta.get("pages") or []) if page})
    page = meta.get("page")
    if page:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    if page and page not in pages:
        pages.append(page)
        pages.sort()
    kind = _locator_kind(source_type, meta)
    available = bool(regions or locator or pages)
    return {
        "schema_version": 1,
        "attribution_level": "chunk",
        "kind": kind,
        "status": "available" if available else "unavailable",
        "pages": pages,
        "locator": locator,
        "regions": regions,
        "regions_truncated": total > len(regions),
    }


def _locator_kind(source_type: str | None, meta: dict[str, Any]) -> str:
    if source_type == "pdf":
        return "pdf"
    if source_type == "image":
        return "image"
    if meta.get("sheet"):
        return "spreadsheet"
    if meta.get("source_line") or meta.get("source_line_start"):
        return "source_lines"
    if meta.get("body_index") is not None:
        return "document_structure"
    if meta.get("row_start") is not None:
        return "table_rows"
    return "section"
