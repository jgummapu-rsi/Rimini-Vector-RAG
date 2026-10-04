"""Normalized provenance -> a human-readable location string for citations.

Loaders attach whatever structural key they have (page/sheet/section/table_index);
this turns any of them into one consistent `location` string carried onto every
chunk and into the vector record, so citations/filters work uniformly.
"""

from __future__ import annotations


def location_str(meta: dict | None) -> str:
    if not meta:
        return ""
    pages = meta.get("pages")
    if pages:
        return f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}-{pages[-1]}"
    if meta.get("page"):
        return f"p.{meta['page']}"
    if meta.get("frame"):
        return f"frame {meta['frame']}"
    if meta.get("sheet"):
        suffix = f", {meta['cell_range']}" if meta.get("cell_range") else ""
        return f"Sheet: {meta['sheet']}{suffix}"
    if meta.get("source_line_start"):
        end = meta.get("source_line_end") or meta["source_line_start"]
        return (
            f"line {end}"
            if end == meta["source_line_start"]
            else f"lines {meta['source_line_start']}-{end}"
        )
    if meta.get("source_line"):
        return f"line {meta['source_line']}"
    if meta.get("body_index") is not None:
        return f"document item {meta['body_index']}"
    if meta.get("row_start") is not None:
        return f"rows {meta['row_start']}-{meta.get('row_end', meta['row_start'])}"
    if meta.get("section"):
        return str(meta["section"]).lstrip("# ").strip()
    if meta.get("section_path"):
        return str(meta["section_path"]).lstrip("# ").strip()
    if meta.get("table_index") is not None:
        return f"table {int(meta['table_index']) + 1}"
    return ""
