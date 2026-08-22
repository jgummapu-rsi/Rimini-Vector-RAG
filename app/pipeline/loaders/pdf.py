"""PDF loader with per-page intelligent routing.

Per page:
  - structured tables (pdfplumber.find_tables) -> deterministic markdown, NO LLM
  - text pages -> text layer (with table regions removed to avoid duplication)
  - scanned pages (little/no text + full image) -> rasterize -> vision OCR
  - embedded figures on a text page -> crop -> vision caption

Pure-Python stack: pdfplumber (text/tables/coords) + pypdfium2 (rasterization).
"""
from __future__ import annotations

import io

from app.domain.models import Modality
from app.pipeline.elements import Element
from app.pipeline.blocks import split_blocks
from app.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT
from app.pipeline.tables import rows_to_markdown

MIN_CHARS_SCANNED = 20        # below this (with an image) => treat page as scanned
MIN_IMG_AREA_PT = 5000.0      # skip icons/rules; points^2
RENDER_SCALE = 200 / 72.0     # ~200 dpi


def extract(data: bytes, filename: str, gateway) -> list[Element]:
    import pdfplumber
    import pypdfium2 as pdfium

    els: list[Element] = []
    order = 0
    render_doc = pdfium.PdfDocument(data)
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pidx, page in enumerate(pdf.pages):
                pno = pidx + 1
                text = (page.extract_text() or "").strip()

                table_bboxes: list[tuple] = []
                try:
                    for t in page.find_tables():
                        md = rows_to_markdown(t.extract())
                        if md and "|" in md:
                            els.append(Element(md, Modality.TABLE.value, "pdf_table",
                                               "structured_table", order, {"page": pno}))
                            order += 1
                            table_bboxes.append(t.bbox)
                except Exception:
                    pass

                images = page.images or []

                if len(text) < MIN_CHARS_SCANNED and images:
                    png = _render_page_png(render_doc, pidx)
                    out = gateway.vision(png, STRICT_TRANSCRIBE_PROMPT, "image/png").strip()
                    if out:
                        new = split_blocks(
                            out, text_extractor="vision", table_extractor="vision_table",
                            text_reason="scanned_page", table_reason="scanned_table",
                            meta={"page": pno})
                        els.extend(new)
                        order += len(new)
                    continue

                body = _text_excluding(page, table_bboxes, text)
                if body.strip():
                    els.append(Element(body, Modality.TEXT.value, "pdf_text",
                                       "text_layer", order, {"page": pno}))
                    order += 1

                for im in images:
                    if _img_area(im) < MIN_IMG_AREA_PT:
                        continue
                    crop = _render_crop_png(render_doc, pidx, im)
                    if crop is None:
                        continue
                    cap = gateway.vision(crop, STRICT_TRANSCRIBE_PROMPT, "image/png").strip()
                    if not cap:
                        continue
                    new = split_blocks(
                        cap, text_extractor="vision", table_extractor="vision_table",
                        text_reason="figure_detected", table_reason="figure_table",
                        text_modality=Modality.IMAGE.value, meta={"page": pno})
                    els.extend(new)
                    order += len(new)
    finally:
        render_doc.close()
    return els


def _img_area(im: dict) -> float:
    return abs(float(im.get("x1", 0) - im.get("x0", 0))) * \
           abs(float(im.get("bottom", 0) - im.get("top", 0)))


def _text_excluding(page, table_bboxes: list[tuple], full_text: str) -> str:
    if not table_bboxes:
        return full_text
    try:
        cropped = page
        for bb in table_bboxes:
            cropped = cropped.outside_bbox(bb)
        return cropped.extract_text() or ""
    except Exception:
        return full_text


def _render_page_png(render_doc, index: int, scale: float = RENDER_SCALE) -> bytes:
    pil = render_doc[index].render(scale=scale).to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _render_crop_png(render_doc, index: int, im: dict, scale: float = RENDER_SCALE):
    try:
        pil = render_doc[index].render(scale=scale).to_pil()
        box = (
            int(float(im["x0"]) * scale), int(float(im["top"]) * scale),
            int(float(im["x1"]) * scale), int(float(im["bottom"]) * scale),
        )
        crop = pil.crop(box)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
