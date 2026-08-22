"""DOCX loader: order-preserving paragraphs + tables (+ embedded images).

Paragraphs and tables are emitted in document order so the chunker keeps their
relationship. Heading styles become markdown headers to aid structure-aware
chunking. Tables -> deterministic markdown (no LLM). Embedded images -> vision.

A level-aware heading STACK (not just the nearest heading) is tracked across
paragraphs, the same mechanism app.pipeline.blocks uses for markdown -- so a
DOCX-sourced chunk gets the same full "Doc Title > H2 > H3" ancestor path
(`meta["section_path"]`) as markdown does, not just the nearest heading
(`meta["section"]`, kept for backward compatibility). chunker.py needs no
changes for this: `_record`/`_reserve_for_prefix` already key off
`section_path` regardless of which loader produced it.
"""
from __future__ import annotations

import io

from app.domain.models import Modality
from app.pipeline.elements import Element
from app.pipeline.blocks import split_blocks
from app.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT
from app.pipeline.tables import rows_to_markdown


def extract(data: bytes, filename: str, gateway) -> list[Element]:
    from docx import Document as Docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Docx(io.BytesIO(data))
    els: list[Element] = []
    order = 0
    heading_stack: list[tuple[int, str]] = []  # [(level, "#.. text"), ...], outermost first

    def path() -> str:
        return " > ".join(h for _, h in heading_stack)

    def nearest() -> str | None:
        return heading_stack[-1][1] if heading_stack else None

    def section_meta() -> dict:
        return {"section": nearest(), "section_path": path()} if heading_stack else {}

    for child in doc.element.body.iterchildren():
        tag = child.tag
        if tag.endswith("}p"):
            para = Paragraph(child, doc)
            t = para.text.strip()
            if not t:
                continue
            style = (para.style.name if para.style else "") or ""
            t = _apply_heading(para, t)
            if style.lower().startswith("heading"):
                level = _heading_level(style)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, t))
            els.append(Element(t, Modality.TEXT.value, "docx", "paragraph", order, section_meta()))
            order += 1
        elif tag.endswith("}tbl"):
            tbl = Table(child, doc)
            rows = [[cell.text for cell in row.cells] for row in tbl.rows]
            md = rows_to_markdown(rows)
            if md:
                meta = section_meta()
                if heading_stack:              # caption repeated on table chunks
                    md = f"{nearest()}\n{md}"
                els.append(Element(md, Modality.TABLE.value, "docx_table",
                                   "structured_table", order, meta))
                order += 1

    order = _extract_images(doc, gateway, els, order)
    return els


def _heading_level(style: str) -> int:
    digits = "".join(ch for ch in style if ch.isdigit())
    level = int(digits) if digits.isdigit() else 2
    return max(1, min(level, 6))


def _apply_heading(para, text: str) -> str:
    style = (para.style.name if para.style else "") or ""
    if style.lower().startswith("heading"):
        return "#" * _heading_level(style) + " " + text
    return text


def _extract_images(doc, gateway, els: list[Element], order: int) -> int:
    from app.gateway.client import GatewayError

    try:
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            blob = rel.target_part.blob
            mime = getattr(rel.target_part, "content_type", None) or "image/png"
            caption = gateway.vision(blob, STRICT_TRANSCRIBE_PROMPT, mime).strip()
            if not caption:
                continue
            new = split_blocks(
                caption, text_extractor="vision", table_extractor="vision_table",
                text_reason="embedded_image", table_reason="embedded_image_table",
                text_modality=Modality.IMAGE.value)
            els.extend(new)
            order += len(new)
    except GatewayError:
        raise
    except Exception:
        pass
    return order
