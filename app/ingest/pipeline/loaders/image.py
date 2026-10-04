"""Standalone image loader: strict LLM transcription, table-aware.

If the image (or a region of it) is a table, it becomes a TABLE element via the
shared block splitter; otherwise a description/transcription as an IMAGE element.
"""

from __future__ import annotations

import io

from PIL import Image, ImageSequence

from app.ingest.pipeline.blocks import split_blocks
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
            buffer = io.BytesIO()
            frame.convert("RGB").save(buffer, format="PNG")
            out = gateway.vision(buffer.getvalue(), STRICT_TRANSCRIBE_PROMPT, "image/png").strip()
            if not out:
                raise UnsafeContentError("Image frame returned no extracted evidence")
            new = split_blocks(
                out,
                text_extractor="vision",
                table_extractor="vision_table",
                text_reason="image_asset",
                table_reason="image_table",
                text_modality=Modality.IMAGE.value,
                meta={"page": index, "frame": index},
            )
            for element in new:
                element.order = len(elements)
                elements.append(element)
    return elements
