"""DOCX loader: order-preserving paragraphs + tables (+ embedded images).

Paragraphs and tables are emitted in document order so the chunker keeps their
relationship. Heading styles become markdown headers to aid structure-aware
chunking. Tables -> deterministic markdown (no LLM). Embedded images -> vision.

A level-aware heading STACK (not just the nearest heading) is tracked across
paragraphs, the same mechanism app.ingest.pipeline.blocks uses for markdown -- so a
DOCX-sourced chunk gets the same full "Doc Title > H2 > H3" ancestor path
(`meta["section_path"]`) as markdown does, not just the nearest heading
(`meta["section"]`, kept for backward compatibility). chunker.py needs no
changes for this: `_record`/`_reserve_for_prefix` already key off
`section_path` regardless of which loader produced it.
"""

from __future__ import annotations

import io
import logging

from docx import Document as Docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.outcomes import omission
from app.ingest.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT
from app.ingest.pipeline.safety import UnsafeContentError, check_image_pixels
from app.ingest.pipeline.tables import rows_to_markdown
from app.shared.domain.models import Modality
from app.shared.gateway.client import GatewayError

log = logging.getLogger(__name__)


def extract(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    """Order-preserving paragraphs+tables, plus best-effort captioned embedded images."""

    doc = Docx(io.BytesIO(data))
    els: list[Element] = []
    order = 0
    heading_stack: list[tuple[int, str]] = []

    def path() -> str:
        return " > ".join(h for _, h in heading_stack)

    def nearest() -> str | None:
        return heading_stack[-1][1] if heading_stack else None

    def section_meta() -> dict:
        return {"section": nearest(), "section_path": path()} if heading_stack else {}

    paragraph_index = 0
    table_index = 0
    for body_index, child in enumerate(doc.element.body.iterchildren(), 1):
        tag = child.tag
        if tag.endswith("}p"):
            para = Paragraph(child, doc)
            t = para.text.strip()
            style = (para.style.name if para.style else "") or ""
            t = _apply_heading(para, t)
            if style.lower().startswith("heading"):
                level = _heading_level(style)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, t))
            if t and not child.xpath(".//a:blip"):
                paragraph_index += 1
                els.append(
                    Element(
                        t,
                        Modality.TEXT.value,
                        "docx",
                        "paragraph",
                        order,
                        {
                            **section_meta(),
                            "body_index": body_index,
                            "paragraph_index": paragraph_index,
                        },
                    )
                )
                order += 1
            if child.xpath(".//a:blip"):
                for run in para.runs:
                    for item in run._r:
                        if item.tag.endswith("}t") and item.text and item.text.strip():
                            paragraph_index += 1
                            els.append(
                                Element(
                                    item.text.strip(),
                                    "text",
                                    "docx",
                                    "paragraph",
                                    order,
                                    {
                                        **section_meta(),
                                        "body_index": body_index,
                                        "paragraph_index": paragraph_index,
                                    },
                                )
                            )
                            order += 1
                        if item.tag.endswith(("}drawing", "}pict")):
                            order = _extract_images(
                                doc,
                                gateway,
                                els,
                                order,
                                cfg,
                                item,
                                {**section_meta(), "body_index": body_index},
                            )
        elif tag.endswith("}tbl"):
            tbl = Table(child, doc)
            rows = [[cell.text for cell in row.cells] for row in tbl.rows]
            md = rows_to_markdown(rows)
            if md:
                table_index += 1
                meta = {**section_meta(), "body_index": body_index, "table_index": table_index - 1}
                if heading_stack:
                    md = f"{nearest()}\n{md}"
                els.append(
                    Element(md, Modality.TABLE.value, "docx_table", "structured_table", order, meta)
                )
                order += 1
            order = _extract_images(doc, gateway, els, order, cfg, child, section_meta())
        for obj in child.xpath(".//w:txbxContent | .//w:object"):
            els.append(
                omission(
                    "docx",
                    "unsupported_embedded_object",
                    order,
                    section_path=path(),
                    object_type=obj.tag.rsplit("}", 1)[-1],
                )
            )
            order += 1

    seen_parts = set()
    for section in doc.sections:
        for part in (section.header, section.footer):
            if part.is_linked_to_previous or str(part.part.partname) in seen_parts:
                continue
            seen_parts.add(str(part.part.partname))
            if any(paragraph.text.strip() for paragraph in part.paragraphs) or part.tables:
                els.append(
                    omission(
                        "docx",
                        "header_footer_not_extracted",
                        order,
                        source_part=str(part.part.partname),
                    )
                )
                order += 1

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


def _extract_images(doc, gateway, els: list[Element], order: int, cfg, anchor, meta) -> int:

    failures = []
    for blip in anchor.xpath(".//a:blip"):
        rel_id = blip.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        )
        try:
            rel = doc.part.rels[rel_id]
            blob = rel.target_part.blob
            mime = getattr(rel.target_part, "content_type", None) or "image/png"
            check_image_pixels(blob, cfg.max_image_pixels)
            caption = gateway.vision(blob, STRICT_TRANSCRIBE_PROMPT, mime).strip()
            if not caption:
                raise UnsafeContentError("DOCX image returned no evidence")
            new = split_blocks(
                caption,
                text_extractor="vision",
                table_extractor="vision_table",
                text_reason="embedded_image",
                table_reason="embedded_image_table",
                text_modality=Modality.IMAGE.value,
                meta={**meta, "relationship_id": rel_id},
            )
            for element in new:
                element.order = order
                order += 1
                els.append(element)
        except GatewayError:
            raise
        except (KeyError, ValueError, OSError, AttributeError) as exc:
            failures.append(type(exc).__name__)
            els.append(
                Element(
                    "",
                    Modality.IMAGE.value,
                    "docx_image",
                    "omitted_image",
                    order,
                    {**meta, "relationship_id": rel_id, "omission": type(exc).__name__},
                )
            )
            order += 1
    if failures:
        log.warning(
            "DOCX anchored images omitted",
            extra={"event": "docx_image_omissions", "omitted_regions": len(failures)},
        )
    return order
