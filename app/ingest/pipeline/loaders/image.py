"""Standalone image loader: strict LLM transcription, table-aware.

If the image (or a region of it) is a table, it becomes a TABLE element via the
shared block splitter; otherwise a description/transcription as an IMAGE element.
"""

from __future__ import annotations

import io
import re

from PIL import Image, ImageOps, ImageSequence

from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.layout import detect_layout
from app.ingest.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT
from app.ingest.pipeline.safety import UnsafeContentError, check_image_pixels
from app.shared.domain.models import Modality


def extract(data: bytes, filename: str, gateway, cfg) -> list:
    """Transcribe a standalone image (or its embedded table) via the vision model."""
    check_image_pixels(data, cfg.max_image_pixels)

    elements = []
    with Image.open(io.BytesIO(data)) as image:
        if getattr(image, "n_frames", 1) > cfg.max_pdf_pages:
            raise UnsafeContentError("Image exceeds the frame limit")
        for index, frame in enumerate(ImageSequence.Iterator(image), 1):
            if frame.width * frame.height > cfg.max_image_pixels:
                raise UnsafeContentError("Image frame exceeds the pixel limit")
            # Use exactly the orientation rendered by the versioned frame endpoint.
            with ImageOps.exif_transpose(frame).convert("RGB") as bitmap:
                png = _png(bitmap)
                geometry = {
                    "page": index,
                    "frame": index,
                    "page_width": bitmap.width,
                    "page_height": bitmap.height,
                    "page_box": (0, 0, bitmap.width, bitmap.height),
                }
                regions = []
                detected = (
                    gateway.layout(png) if hasattr(gateway, "layout") else detect_layout(png, cfg)
                )
                for region in detected:
                    x0, y0, x1, y1 = region["bbox"]
                    bbox = (
                        max(0, round(x0 * bitmap.width)),
                        max(0, round(y0 * bitmap.height)),
                        min(bitmap.width, round(x1 * bitmap.width)),
                        min(bitmap.height, round(y1 * bitmap.height)),
                    )
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    with bitmap.crop(bbox) as crop:
                        text = gateway.vision(
                            _png(crop), STRICT_TRANSCRIBE_PROMPT, "image/png"
                        ).strip()
                    if text:
                        regions.append({**region, "bbox": bbox, "text": text})
                # Retain the full-frame transcription even when layout misses text.
                out = gateway.vision(png, STRICT_TRANSCRIBE_PROMPT, "image/png").strip()
            if not out:
                raise UnsafeContentError("Image frame returned no extracted evidence")
            new = split_blocks(
                out,
                text_extractor="vision",
                table_extractor="vision_table",
                text_reason="image_asset",
                table_reason="image_table",
                text_modality=Modality.IMAGE.value,
                meta=geometry,
            )
            for element in new:
                element.order = len(elements)
                element.meta["source_spans"] = _source_spans(
                    element.text, regions, geometry, element.order, out
                )
                elements.append(element)
    return elements


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _normalized(text: str) -> tuple[str, list[int]]:
    """Normalize formatting, retaining punctuation, word boundaries and offsets."""
    chars, offsets = [], []
    for index, char in enumerate(text):
        if char in "*`_#":
            continue
        value = " " if char.isspace() else char.casefold()
        if value == " " and chars and chars[-1] == " ":
            continue
        chars.extend(value)
        offsets.extend([index] * len(value))
    return "".join(chars), offsets


def _matches(text: str, needle: str) -> list:
    return list(re.finditer(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text)) if needle else []


def _source_spans(
    text: str, regions: list[dict], geometry: dict, element: int, frame_text: str
) -> list[dict]:
    """Locate uniquely matching crop transcriptions; never infer word rectangles.

    Unmatched text retains frame geometry. Overlapping or repeated matches are
    deliberately left at frame precision rather than choosing an arbitrary crop.
    """
    normalized, offsets = _normalized(text)
    full_text = _normalized(frame_text)[0]
    candidates = []
    for region in regions:
        value = _normalized(region["text"])[0].strip()
        matches = _matches(normalized, value)
        if len(matches) == 1 and len(_matches(full_text, value)) == 1:
            match = matches[0]
            candidates.append((offsets[match.start()], offsets[match.end() - 1] + 1, region))
        elif (
            not matches
            and len(_matches(value, normalized.strip())) == 1
            and len(_matches(full_text, normalized.strip())) == 1
        ):
            # A table/heading split can occupy only part of a detected block.
            candidates.append((0, len(text), region))
    selected = [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate and candidate[0] < other[1] and other[0] < candidate[1]
            for other in candidates
        )
    ]
    selected.sort(key=lambda candidate: candidate[0])
    spans = []

    def append(start, end, region=None):
        if not text[start:end].strip():
            return
        span = {
            **geometry,
            "element": element,
            "start": start,
            "end": end,
            "text": text[start:end],
            "bbox": region["bbox"] if region else geometry["page_box"],
            "precision": "element_region" if region else "page_region",
        }
        if region:
            span.update(
                layout_id=region["id"],
                layout_label=region["label"],
                layout_confidence=region["confidence"],
            )
        spans.append(span)

    cursor = 0
    for start, end, region in selected:
        append(cursor, start)
        append(start, end, region)
        cursor = end
    append(cursor, len(text))
    return spans
