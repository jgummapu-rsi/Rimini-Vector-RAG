from __future__ import annotations

from app.ingest.pipeline.elements import Element


def omission(extractor: str, reason: str, order: int, **source) -> Element:
    return Element(
        "", "image", extractor, reason, order, {**source, "omission": reason, "outcome": "omitted"}
    )
