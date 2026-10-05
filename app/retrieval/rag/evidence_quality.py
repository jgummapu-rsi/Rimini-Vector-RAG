"""Conservative, query-aware filtering of PDF page furniture.

Keep source text and identifiers intact. Repeated body prose may belong to
different subjects and must not be collapsed just because its text matches.
"""

from __future__ import annotations

import re
from collections import defaultdict


def useful_indices(question: str, texts: list[str], metadata: list[dict]) -> list[int]:
    if re.search(
        r"\b(logo|branding|font|header|footer|author|approver|revision|release date|copyright|table of contents)\b",
        question,
        re.I,
    ):
        return list(range(len(texts)))
    repeated = defaultdict(list)
    excluded = set()
    for i, (text, meta) in enumerate(zip(texts, metadata, strict=True)):
        section = meta.get("section_path")
        body = text.partition("\n\n")[2] if section and text.startswith(section + "\n\n") else text
        provenance = meta.get("provenance") or {}
        regions = provenance.get("regions") or []
        if not regions or provenance.get("regions_truncated"):
            continue
        # A mixed body/footer chunk remains intact for citation correctness.
        if all(r.get("layout_label") == "abandon" for r in regions):
            excluded.add(i)
            continue
        if len(body.split()) < 180 and re.search(
            r"\b(logo/wordmark|company logo|logo visible)\b", body, re.I
        ):
            if all(r.get("y", 1) + r.get("height", 1) < 0.2 for r in regions):
                excluded.add(i)
                continue
        # Require identical text on distinct pages within the same document.
        if len(body.split()) < 100 and all(
            r.get("y", 1) + r.get("height", 1) < 0.23 for r in regions
        ):
            key = (
                meta.get("document_id", meta.get("_id")),
                meta.get("generation_id"),
                " ".join(body.split()),
            )
            repeated[key].append((i, tuple(provenance.get("pages") or [])))
    for group in repeated.values():
        if len({pages for _, pages in group if pages}) >= 2:
            excluded.update(i for i, _ in group)
    return [i for i in range(len(texts)) if i not in excluded]
