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
    if meta.get("sheet"):
        return f"Sheet: {meta['sheet']}"
    if meta.get("section"):
        return str(meta["section"]).lstrip("# ").strip()
    if meta.get("table_index") is not None:
        return f"table {int(meta['table_index']) + 1}"
    return ""
