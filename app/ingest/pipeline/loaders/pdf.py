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
import logging
import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import median

import pdfplumber
import pypdfium2 as pdfium
from PIL import ImageDraw

from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.layout import detect_layout
from app.ingest.pipeline.outcomes import omission
from app.ingest.pipeline.prompts import FIGURE_EXTRACTION_PROMPT, STRICT_TRANSCRIBE_PROMPT
from app.ingest.pipeline.safety import UnsafeContentError
from app.ingest.pipeline.tables import rows_to_markdown
from app.shared.domain.models import Modality

MIN_CHARS_SCANNED = 20
RENDER_SCALE = 200 / 72.0
MIN_IMAGE_SIDE_PX = 24
_CID_TOKEN_RE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
log = logging.getLogger(__name__)


def extract(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    """Per-page: tables -> markdown, scanned pages -> OCR, figures -> captioned."""

    els: list[Element] = []
    order = 0
    render_doc = pdfium.PdfDocument(data)
    try:
        if len(render_doc) > cfg.max_pdf_pages:
            raise UnsafeContentError(
                f"PDF has {len(render_doc)} pages, exceeds the {cfg.max_pdf_pages} page limit"
            )
        outline = _outline_sections(render_doc)
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pidx, page in enumerate(pdf.pages):
                pno = pidx + 1
                page_width = float(getattr(page, "width", 0.0))
                page_height = float(getattr(page, "height", 0.0))
                page_geometry = (
                    {"page_width": page_width, "page_height": page_height}
                    if page_width > 0 and page_height > 0
                    else {}
                )
                page_geometry.update(outline.get(pno, {}))
                text = (page.extract_text() or "").strip()
                page_elements: list[tuple[tuple[float, float], Element]] = []
                layout_regions = []
                if getattr(cfg, "layout_enabled", False):
                    png = _render_page_png(render_doc, pidx, max_pixels=cfg.max_image_pixels)
                    if hasattr(gateway, "layout"):
                        detected = gateway.layout(png)
                    else:
                        detected = detect_layout(png, cfg)
                    # PDFium renders the CropBox; pdfplumber positions are in
                    # its rotated MediaBox coordinate system.
                    crop = tuple(
                        getattr(page, "cropbox", None)
                        or getattr(page, "bbox", (0, 0, page_width, page_height))
                    )
                    layout_regions = [
                        dict(
                            region,
                            bbox=(
                                crop[0] + region["bbox"][0] * (crop[2] - crop[0]),
                                crop[1] + region["bbox"][1] * (crop[3] - crop[1]),
                                crop[0] + region["bbox"][2] * (crop[2] - crop[0]),
                                crop[1] + region["bbox"][3] * (crop[3] - crop[1]),
                            ),
                        )
                        for region in detected
                    ]
                page_geometry["page_box"] = tuple(
                    getattr(page, "cropbox", None)
                    or getattr(page, "bbox", (0, 0, page_width, page_height))
                )

                table_bboxes: list[tuple] = []
                try:
                    tables = page.find_tables()
                except Exception as exc:  # noqa: BLE001 - parser failures are page-local
                    log.warning(
                        "PDF table detection failed",
                        extra={
                            "event": "pdf_table_detection_failed",
                            "page": pno,
                            "error_type": type(exc).__name__,
                        },
                    )
                    page_elements.append(
                        ((0.0, 0.0), omission("pdf_table", "table_detection_failed", 0, page=pno))
                    )
                    tables = []
                for table_index, table in enumerate(tables):
                    try:
                        rows = table.extract()
                        md = rows_to_markdown(rows)
                        bbox = tuple(table.bbox)
                    except Exception as exc:  # noqa: BLE001 - preserve remaining tables/page text
                        log.warning(
                            "PDF table extraction failed",
                            extra={
                                "event": "pdf_table_extraction_failed",
                                "page": pno,
                                "table_index": table_index,
                                "error_type": type(exc).__name__,
                            },
                        )
                        page_elements.append(
                            (
                                (0.0, 0.0),
                                omission(
                                    "pdf_table",
                                    "table_extraction_failed",
                                    0,
                                    page=pno,
                                    table_index=table_index,
                                ),
                            )
                        )
                        continue
                    if md and "|" in md:
                        meta = {
                            "page": pno,
                            "table_index": table_index,
                            "bbox": bbox,
                            **page_geometry,
                        }
                        page_elements.append(
                            (
                                _bbox_position(bbox),
                                Element(
                                    md,
                                    Modality.TABLE.value,
                                    "pdf_table",
                                    "structured_table",
                                    0,
                                    meta,
                                ),
                            )
                        )
                        table_bboxes.append(bbox)

                images = page.images or []
                text_status = _native_text_status(text)
                has_unresolved_visuals = bool(images) or _has_vector_graphics(page)
                needs_full_page_vision = text_status == "corrupt" or (
                    text_status == "sparse" and has_unresolved_visuals and not table_bboxes
                )

                if needs_full_page_vision:
                    visual_regions = []
                    # Associate existing vision extraction with detected crops.
                    # This supplies block geometry, never invented word boxes.
                    if layout_regions:
                        bitmap = _render_page(render_doc, pidx, max_pixels=cfg.max_image_pixels)
                        for region in layout_regions:
                            crop = _layout_crop(bitmap, region["bbox"], page_geometry["page_box"])
                            out = gateway.vision(
                                crop, STRICT_TRANSCRIBE_PROMPT, "image/png"
                            ).strip()
                            if not out:
                                continue
                            visual_regions.append((out, region))
                        # Full-page extraction remains the completeness fallback;
                        # uncovered text must not disappear due to detector misses.
                    png = _render_page_png(render_doc, pidx, max_pixels=cfg.max_image_pixels)
                    out = gateway.vision(png, STRICT_TRANSCRIBE_PROMPT, "image/png").strip()
                    if out:
                        text_reason = (
                            "untrusted_text_layer"
                            if text_status == "corrupt"
                            else "visual_page_without_text_layer"
                            if not images
                            else "scanned_page"
                        )
                        table_reason = (
                            "untrusted_text_layer_table"
                            if text_status == "corrupt"
                            else "visual_page_without_text_layer_table"
                            if not images
                            else "scanned_table"
                        )
                        new = split_blocks(
                            out,
                            text_extractor="vision",
                            table_extractor="vision_table",
                            text_reason=text_reason,
                            table_reason=table_reason,
                            meta={"page": pno, **page_geometry},
                        )
                        native_regions = _native_text_regions(page, [], text)
                        for element in new:
                            spans = _matching_region_spans(
                                element.text, native_regions, order, pno, page_geometry
                            )
                            if not spans:
                                compact = _compact_char_offsets(element.text)[0]
                                matches = [
                                    region
                                    for value, region in visual_regions
                                    if compact and compact in _compact_char_offsets(value)[0]
                                ]
                                if len(matches) == 1:
                                    region = matches[0]
                                    spans = [
                                        {
                                            "element": order,
                                            "start": 0,
                                            "end": len(element.text),
                                            "page": pno,
                                            "bbox": region["bbox"],
                                            "text": element.text,
                                            **page_geometry,
                                            **_layout_meta(region),
                                            "precision": "element_region",
                                        }
                                    ]
                            if not spans and page_geometry:
                                spans = [
                                    {
                                        "element": order,
                                        "start": 0,
                                        "end": len(element.text),
                                        "page": pno,
                                        "bbox": (0.0, 0.0, page_width, page_height),
                                        "text": element.text,
                                        **page_geometry,
                                        "precision": "page_region",
                                    }
                                ]
                            if spans:
                                element.meta["source_spans"] = spans
                            element.order = order
                            order += 1
                            els.append(element)
                    else:
                        raise UnsafeContentError("Nonblank PDF page returned no extracted evidence")
                    continue

                if text_status == "usable" and _has_vector_graphics(page) and not table_bboxes:
                    page_elements.append(
                        (
                            (0.0, 0.0),
                            omission("pdf_graphics", "vector_graphics_not_extracted", 0, page=pno),
                        )
                    )

                for body, bbox, words, region in _native_word_regions(
                    page, table_bboxes, text, layout_regions
                ):
                    offset = 0
                    spans = []
                    for word in words:
                        value = str(word["text"])
                        spans.append(
                            {
                                "start": offset,
                                "end": offset + len(value),
                                "text": value,
                                "page": pno,
                                "bbox": (word["x0"], word["top"], word["x1"], word["bottom"]),
                                **page_geometry,
                                **_layout_meta(_word_layout(word, layout_regions)),
                                "precision": "word",
                            }
                        )
                        offset += len(value) + 1
                    page_elements.append(
                        (
                            _bbox_position(bbox),
                            Element(
                                body,
                                Modality.TEXT.value,
                                "pdf_text",
                                "text_layer",
                                0,
                                {
                                    "page": pno,
                                    "bbox": bbox,
                                    **page_geometry,
                                    **_layout_meta(region),
                                    **_typography(words),
                                    **(
                                        {"source_spans": spans}
                                        if spans
                                        else {"precision": "page_region"}
                                    ),
                                },
                            ),
                        )
                    )

                page_bitmap = None
                page_bitmap_failed = False
                figure_regions = [
                    region for region in layout_regions if region["label"] == "figure"
                ]
                for region in figure_regions:
                    if page_bitmap is None:
                        page_bitmap = _render_page(
                            render_doc, pidx, RENDER_SCALE, cfg.max_image_pixels
                        )
                    crop = _layout_crop(page_bitmap, region["bbox"], page_geometry["page_box"])
                    cap = gateway.vision(crop, FIGURE_EXTRACTION_PROMPT, "image/png").strip()
                    if not cap:
                        page_elements.append(
                            (
                                _bbox_position(region["bbox"]),
                                omission(
                                    "pdf_image",
                                    "empty_figure_extraction",
                                    0,
                                    page=pno,
                                    bbox=region["bbox"],
                                ),
                            )
                        )
                        continue
                    new = split_blocks(
                        cap,
                        text_extractor="vision",
                        table_extractor="vision_table",
                        text_reason="layout_figure",
                        table_reason="figure_table",
                        text_modality=Modality.IMAGE.value,
                        meta={
                            "page": pno,
                            "bbox": region["bbox"],
                            **page_geometry,
                            **_layout_meta(region),
                            "precision": "element_region",
                        },
                    )
                    page_elements.extend(
                        (_bbox_position(region["bbox"]), element) for element in new
                    )
                for im in images:
                    if _bbox_coverage(im, [region["bbox"] for region in figure_regions]) >= 0.9:
                        continue
                    if not _image_is_readable(im, RENDER_SCALE):
                        page_elements.append(
                            (
                                (float(im.get("top", 0)), float(im.get("x0", 0))),
                                omission("pdf_image", "image_below_resolution_limit", 0, page=pno),
                            )
                        )
                        continue
                    if text_status == "usable" and _image_backs_native_text(page, im):
                        continue
                    if _bbox_coverage(im, table_bboxes) >= 0.95:
                        continue
                    if page_bitmap is None and not page_bitmap_failed:
                        try:
                            page_bitmap = _render_page(
                                render_doc, pidx, RENDER_SCALE, cfg.max_image_pixels
                            )
                        except UnsafeContentError:
                            page_bitmap_failed = True
                    if page_bitmap is None:
                        page_elements.append(
                            (
                                _bbox_position(_image_bbox(im)),
                                omission(
                                    "pdf_image",
                                    "page_render_limit",
                                    0,
                                    page=pno,
                                    bbox=_image_bbox(im),
                                ),
                            )
                        )
                        continue
                    crop = _crop_png(page_bitmap, im, RENDER_SCALE, excluded_bboxes=table_bboxes)
                    if crop is None:
                        page_elements.append(
                            (
                                _bbox_position(_image_bbox(im)),
                                omission(
                                    "pdf_image",
                                    "image_crop_failed",
                                    0,
                                    page=pno,
                                    bbox=_image_bbox(im),
                                ),
                            )
                        )
                        continue
                    cap = gateway.vision(crop, FIGURE_EXTRACTION_PROMPT, "image/png").strip()
                    if not cap:
                        page_elements.append(
                            (
                                _bbox_position(_image_bbox(im)),
                                omission(
                                    "pdf_image",
                                    "empty_figure_extraction",
                                    0,
                                    page=pno,
                                    bbox=_image_bbox(im),
                                ),
                            )
                        )
                        continue
                    new = split_blocks(
                        cap,
                        text_extractor="vision",
                        table_extractor="vision_table",
                        text_reason="figure_description",
                        table_reason="figure_table",
                        text_modality=Modality.IMAGE.value,
                        meta={"page": pno, "bbox": _image_bbox(im), **page_geometry},
                    )
                    position = _bbox_position(_image_bbox(im))
                    page_elements.extend((position, element) for element in new)

                for _, element in _reading_order(page_elements, float(getattr(page, "width", 0))):
                    element.order = order
                    for span in element.meta.get("source_spans", []):
                        span["element"] = order
                    order += 1
                    els.append(element)
    finally:
        render_doc.close()
    _annotate_sections(els)
    return els


def _typography(words):
    sizes = [float(word["size"]) for word in words if word.get("size")]
    return (
        {
            "font_size": median(sizes),
            "font_bold": sum(
                len(str(word["text"]))
                for word in words
                if re.search(r"bold|black|demi|heavy", str(word.get("fontname", "")), re.I)
            )
            >= sum(len(str(word["text"])) for word in words) * 0.8,
        }
        if sizes
        else {}
    )


def _annotate_sections(elements):
    """Carry visible heading groups into their details and page continuations.

    Typography and layout are signals, not a vocabulary of document subjects.
    Same-baseline right-hand labels stay with the heading group. The heading
    text is context; its geometry is never attached to continuation evidence.
    """
    sizes = [
        e.meta["font_size"]
        for e in elements
        if e.extractor == "pdf_text" and len(e.text.split()) >= 10 and e.meta.get("font_size")
    ]
    body_size = median(sizes) if sizes else 0
    major = ""
    heading = []
    pending = []
    active = ""
    previous_outline = None

    def finish():
        nonlocal active, heading, pending
        if heading:
            active = " > ".join(dict.fromkeys(filter(None, [major, *heading])))
        for item in pending:
            if active:
                outline = item.meta.get("section_path", "")
                item.meta["section_path"] = " > ".join(filter(None, [outline, active]))
                item.meta["section_source"] = "pdf_visible_heading"
        pending, heading = [], []

    for element in elements:
        if element.meta.get("omission"):
            continue
        outline = element.meta.get("section_path")
        if outline != previous_outline:
            finish()
            major, active = "", ""
            previous_outline = outline
        text = element.text.strip()
        box = element.meta.get("bbox")
        width = element.meta.get("page_width", 0)
        short = (
            0 < len(text.split()) <= 18
            and not re.match(r"^[●•▪*-]\s", text)
            and text.count(",") < 2
            and ":" not in text
        )
        styled = element.meta.get("layout_label") == "title" or (
            body_size
            and element.meta.get("font_size", 0)
            >= body_size * (1.05 if element.meta.get("font_bold") else 1.08)
        )
        candidate = (
            element.extractor == "pdf_text"
            and short
            and styled
            and not text.endswith((".", ":", ";", "!", "?"))
            and box
            and box[0] < width * 0.65
        )
        if candidate:
            # A new heading after detail starts a new scope. Adjacent title /
            # subtitle lines form one scope even with dates on their right.
            if (
                heading
                and box
                and pending
                and element.meta.get("page") == pending[-1].meta.get("page")
                and (box[1] - pending[-1].meta["bbox"][3] > max(8, (box[3] - box[1]) * 0.75))
            ):
                finish()
            if not heading:
                finish()
            letters = "".join(char for char in text.split("(", 1)[0] if char.isalpha())
            if letters.isupper() and len(letters) >= 4 and box[0] > width * 0.25:
                finish()
                major, active = text, ""
            heading.append(text)
            element.meta["block_type"] = "heading"
            pending.append(element)
        elif (
            heading
            and box
            and pending
            and element.meta.get("page") == pending[-1].meta.get("page")
            and (box[0] > width * 0.65 and box[1] <= pending[-1].meta["bbox"][3])
        ):
            pending.append(element)
        else:
            finish()
            if active:
                element.meta["section_path"] = " > ".join(filter(None, [outline, active]))
                element.meta["section_source"] = "pdf_visible_heading"
    finish()


def _outline_sections(document) -> dict[int, dict]:
    """Carry explicit PDF bookmark hierarchy across its page ranges.

    Bookmarks are source-provided boundaries, not inferred entities. They apply
    equally to chapters, products, contracts and bundled documents. Ambiguous
    same-page destinations are omitted rather than attributing a whole page to
    whichever bookmark happened to appear last.
    """

    starts = defaultdict(list)
    stack = []
    try:
        for bookmark in document.get_toc():
            title = bookmark.get_title().strip()
            level = bookmark.level
            while stack and stack[-1][0] >= level:
                stack.pop()
            if title:
                stack.append((level, title))
            destination = bookmark.get_dest()
            index = destination.get_index() if destination is not None else None
            if title and index is not None and 0 <= index < len(document):
                starts[index + 1].append(" > ".join(value for _, value in stack))
    except Exception:
        log.warning("PDF outline unavailable", extra={"event": "pdf_outline_failed"})
        return {}
    output = {}
    boundaries = sorted(starts)
    for position, start in enumerate(boundaries):
        end = boundaries[position + 1] - 1 if position + 1 < len(boundaries) else len(document)
        paths = list(dict.fromkeys(starts[start]))
        # Parent and child bookmarks commonly share a start page; the child
        # is the most specific source path. Unrelated siblings are ambiguous.
        path = max(paths, key=len)
        if any(value != path and not path.startswith(value + " > ") for value in paths):
            continue
        for page in range(start, end + 1):
            output[page] = {
                "section_path": path,
                "section_start_page": start,
                "section_end_page": end,
                "section_source": "pdf_outline",
            }
    return output


def _native_text_status(text: str) -> str:
    """Classify native PDF text using only high-confidence corruption signals."""
    chars = [ch for ch in text if not ch.isspace()]
    suspicious = sum(ch == "\ufffd" or unicodedata.category(ch).startswith("C") for ch in chars)
    if suspicious >= max(2, math.ceil(len(chars) * 0.10)):
        return "corrupt"
    if len(_CID_TOKEN_RE.findall(text)) >= 2:
        return "corrupt"
    if len(chars) < MIN_CHARS_SCANNED:
        return "sparse"
    return "usable"


def _matching_regions(text: str, regions: list[tuple[str, tuple]]) -> list[tuple[str, tuple]]:
    """Align vision blocks to native PDF lines when a damaged text layer still has geometry."""
    matched = _matching_region_spans(text, regions, 0, 1, {})
    boxes = {tuple(span["bbox"]) for span in matched}
    return [(line, bbox) for line, bbox in regions if tuple(bbox) in boxes]


def _matching_region_spans(
    text: str, regions: list[tuple[str, tuple]], element: int, page: int, page_geometry: dict
) -> list[dict]:
    """Map native PDF line boxes to exact character ranges in vision text.

    Ordered token windows prevent an unrelated line elsewhere on the page from
    matching merely because it shares common words with a vision section.
    """
    evidence = _match_token_offsets(text)
    if not evidence:
        return []
    values = [token for token, _, _ in evidence]
    compact, compact_offsets = _compact_char_offsets(text)
    used: set[int] = set()
    used_chars: set[int] = set()
    spans = []
    for line, bbox in regions:
        expected_compact, _ = _compact_char_offsets(line.rstrip("-\u2010\u2011\u2012\u2013\u2014"))
        compact_start = compact.find(expected_compact) if len(expected_compact) >= 4 else -1
        if compact_start >= 0:
            compact_end = compact_start + len(expected_compact)
            if any(index in used_chars for index in range(compact_start, compact_end)):
                compact_start = compact.find(expected_compact, compact_start + 1)
                compact_end = compact_start + len(expected_compact)
            if compact_start >= 0 and not any(
                index in used_chars for index in range(compact_start, compact_end)
            ):
                start = compact_offsets[compact_start][0]
                end = compact_offsets[compact_end - 1][1]
                used_chars.update(range(compact_start, compact_end))
                spans.append(
                    {
                        "element": element,
                        "start": start,
                        "end": end,
                        "page": page,
                        "bbox": bbox,
                        "text": text[start:end],
                        **page_geometry,
                        "precision": "exact_text",
                    }
                )
                continue
        line_tokens = _match_tokens(line)
        match = _best_token_window(values, line_tokens, used)
        if match is None:
            continue
        first, last = match
        used.update(range(first, last))
        start, end = evidence[first][1], evidence[last - 1][2]
        spans.append(
            {
                "element": element,
                "start": start,
                "end": end,
                "page": page,
                "bbox": bbox,
                "text": text[start:end],
                **page_geometry,
                "precision": "exact_text",
            }
        )
    return spans


def _best_token_window(
    evidence: list[str], expected: list[str], used: set[int]
) -> tuple[int, int] | None:
    if not expected:
        return None
    if len(expected) == 1:
        positions = [index for index, token in enumerate(evidence) if token == expected[0]]
        return (
            (positions[0], positions[0] + 1)
            if len(positions) == 1 and positions[0] not in used
            else None
        )

    best = None
    for size in range(max(2, len(expected) - 2), min(len(evidence), len(expected) + 2) + 1):
        for start in range(len(evidence) - size + 1):
            candidate = (start, start + size)
            if any(index in used for index in range(*candidate)):
                continue
            window = evidence[start : start + size]
            matcher = SequenceMatcher(None, expected, window, autojunk=False)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            coverage = matched / len(expected)
            score = matcher.ratio()
            rank = (score, coverage, matched, -abs(size - len(expected)), -start)
            if (
                matched >= 2
                and coverage >= 0.65
                and score >= 0.65
                and (best is None or rank > best[0])
            ):
                best = (rank, candidate)
    return best[1] if best else None


def _match_tokens(text: str) -> list[str]:
    return [token for token, _, _ in _match_token_offsets(text)]


def _match_token_offsets(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group().casefold(), match.start(), match.end())
        for match in re.finditer(r"[\w@.+%-]+", text)
        if len(match.group()) > 1 and not match.group().casefold().startswith("cid")
    ]


def _compact_char_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars = []
    offsets = []
    for index, char in enumerate(text):
        if char.isalnum():
            folded = char.casefold()
            chars.extend(folded)
            offsets.extend([(index, index + 1)] * len(folded))
    return "".join(chars), offsets


def _has_vector_graphics(page) -> bool:
    """Return whether the page contains drawing objects not represented as images."""
    objects = getattr(page, "objects", None) or {}
    return any(objects.get(kind) for kind in ("line", "curve", "rect"))


def _image_is_readable(im: dict, scale: float = RENDER_SCALE) -> bool:
    try:
        width = abs(float(im["x1"]) - float(im["x0"])) * scale
        height = abs(float(im["bottom"]) - float(im["top"])) * scale
    except (KeyError, TypeError, ValueError):
        return False
    return width >= MIN_IMAGE_SIDE_PX and height >= MIN_IMAGE_SIDE_PX


def _image_backs_native_text(page, im: dict) -> bool:
    """Detect a page-sized scan already represented by a trusted OCR text layer."""
    try:
        page_width = float(page.width)
        page_height = float(page.height)
        x0, x1 = sorted((float(im["x0"]), float(im["x1"])))
        top, bottom = sorted((float(im["top"]), float(im["bottom"])))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if page_width <= 0 or page_height <= 0:
        return False

    left = max(0.0, x0)
    right = min(page_width, x1)
    upper = max(0.0, top)
    lower = min(page_height, bottom)
    if right <= left or lower <= upper:
        return False
    if ((right - left) * (lower - upper)) / (page_width * page_height) < 0.5:
        return False

    chars = getattr(page, "chars", None) or []
    if not chars:
        return False
    covered = 0
    considered = 0
    for char in chars:
        try:
            cx = (float(char["x0"]) + float(char["x1"])) / 2
            cy = (float(char["top"]) + float(char["bottom"])) / 2
        except (KeyError, TypeError, ValueError):
            continue
        considered += 1
        if left <= cx <= right and upper <= cy <= lower:
            covered += 1
    return considered > 0 and covered / considered >= 0.9


def _bbox_coverage(im: dict, bboxes: list[tuple]) -> float:
    try:
        x0, x1 = sorted((float(im["x0"]), float(im["x1"])))
        top, bottom = sorted((float(im["top"]), float(im["bottom"])))
    except (KeyError, TypeError, ValueError):
        return 0.0
    area = (x1 - x0) * (bottom - top)
    if area <= 0:
        return 0.0

    covered = 0.0
    for bbox in bboxes:
        try:
            bx0, btop, bx1, bbottom = map(float, bbox)
        except (TypeError, ValueError):
            continue
        width = max(0.0, min(x1, bx1) - max(x0, bx0))
        height = max(0.0, min(bottom, bbottom) - max(top, btop))
        covered += width * height
    return min(1.0, covered / area)


def _image_bbox(im: dict) -> tuple[float, float, float, float]:
    return (
        float(im["x0"]),
        float(im["top"]),
        float(im["x1"]),
        float(im["bottom"]),
    )


def _bbox_position(bbox: tuple) -> tuple[float, float]:
    return float(bbox[1]), float(bbox[0])


def _reading_order(elements, width: float):
    midpoint = width / 2
    left = [item for item in elements if item[1].meta.get("bbox", (0, 0, width, 0))[2] < midpoint]
    right = [item for item in elements if item[1].meta.get("bbox", (0, 0, width, 0))[0] > midpoint]
    if len(left) < 2 or len(right) < 2:
        return _row_order(elements)
    spanning = [item for item in elements if item not in left and item not in right]
    ordered = []
    pending = left + right
    prose_spans = any(len(item[1].text.split()) > 12 for item in spanning)

    def order_band(band):
        if not band:
            return []
        boxes = [item[1].meta["bbox"] for item in band]
        height = median(box[3] - box[1] for box in boxes)
        depth = max(box[3] for box in boxes) - min(box[1] for box in boxes)
        # Short aligned labels around full-width prose are rows (e.g. title /
        # date, product / price), not two independent newspaper columns.
        if prose_spans and depth < height * 6:
            return _row_order(band)
        return _row_order([item for item in band if item in left]) + _row_order(
            [item for item in band if item in right]
        )

    for divider in sorted(spanning, key=lambda item: item[0]):
        above = [item for item in pending if item[0][0] < divider[0][0]]
        ordered.extend(order_band(above))
        pending = [item for item in pending if item not in above]
        ordered.append(divider)
    ordered.extend(order_band(pending))
    return ordered


def _row_order(elements):
    rows = []
    for item in sorted(elements, key=lambda item: item[0]):
        if not rows or item[0][0] - rows[-1][0][0][0] > 3:
            rows.append([item])
        else:
            rows[-1].append(item)
    return [item for row in rows for item in sorted(row, key=lambda item: item[0][1])]


def _native_text_regions(page, excluded_bboxes: list[tuple], fallback: str):
    return [
        (text, bbox) for text, bbox, _, _ in _native_word_regions(page, excluded_bboxes, fallback)
    ]


def _native_word_regions(page, excluded_bboxes: list[tuple], fallback: str, layout_regions=()):
    try:
        try:
            words = page.extract_words(extra_attrs=["fontname", "size"]) or []
        except TypeError:  # compatible parser implementations without extra_attrs
            words = page.extract_words() or []
    except Exception:  # noqa: BLE001 - extract_text remains the compatibility fallback
        words = []
    if not words:
        body = _text_excluding(page, excluded_bboxes, fallback).strip()
        width = float(getattr(page, "width", 0.0))
        height = float(getattr(page, "height", 0.0))
        return [(body, (0.0, 0.0, width, height), [], None)] if body else []

    words = [
        word
        for word in words
        if not any(
            _point_in_bbox(
                (float(word["x0"]) + float(word["x1"])) / 2,
                (float(word["top"]) + float(word["bottom"])) / 2,
                bbox,
            )
            for bbox in excluded_bboxes
        )
    ]
    if not words:
        return []

    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not lines or abs(float(word["top"]) - float(lines[-1][0]["top"])) > 3:
            lines.append([word])
        else:
            lines[-1].append(word)

    separated_lines = []
    for line in lines:
        line.sort(key=lambda item: float(item["x0"]))
        group = []
        for word in line:
            if group and float(word["x0"]) - float(group[-1]["x1"]) > 36:
                separated_lines.append(group)
                group = []
            group.append(word)
        separated_lines.append(group)
    regions = []
    for line in separated_lines:
        value = " ".join(str(item["text"]) for item in line).strip()
        if value:
            regions.append(
                (
                    value,
                    (
                        min(float(item["x0"]) for item in line),
                        min(float(item["top"]) for item in line),
                        max(float(item["x1"]) for item in line),
                        max(float(item["bottom"]) for item in line),
                    ),
                    line,
                    _word_layout(line[0], layout_regions),
                )
            )
    return regions


def _word_layout(word, regions):
    x = (float(word["x0"]) + float(word["x1"])) / 2
    y = (float(word["top"]) + float(word["bottom"])) / 2
    candidates = [region for region in regions if _point_in_bbox(x, y, region["bbox"])]
    return min(
        candidates,
        key=lambda region: (
            (region["bbox"][2] - region["bbox"][0]) * (region["bbox"][3] - region["bbox"][1])
        ),
        default=None,
    )


def _layout_meta(region):
    return (
        {
            "layout_id": region["id"],
            "layout_label": region["label"],
            "layout_confidence": region["confidence"],
        }
        if region
        else {}
    )


def _layout_crop(bitmap, bbox, page_box):
    left, top, right, bottom = page_box
    sx, sy = bitmap.width / (right - left), bitmap.height / (bottom - top)
    box = (
        max(0, math.floor((bbox[0] - left) * sx)),
        max(0, math.floor((bbox[1] - top) * sy)),
        min(bitmap.width, math.ceil((bbox[2] - left) * sx)),
        min(bitmap.height, math.ceil((bbox[3] - top) * sy)),
    )
    with bitmap.crop(box) as crop:
        output = io.BytesIO()
        crop.save(output, "PNG")
        return output.getvalue()


def _point_in_bbox(x: float, y: float, bbox: tuple) -> bool:
    try:
        x0, top, x1, bottom = map(float, bbox)
    except (TypeError, ValueError):
        return False
    return min(x0, x1) <= x <= max(x0, x1) and min(top, bottom) <= y <= max(top, bottom)


def _text_excluding(page, table_bboxes: list[tuple], full_text: str) -> str:
    if not table_bboxes:
        return full_text
    try:
        cropped = page
        for bb in table_bboxes:
            cropped = cropped.outside_bbox(bb)
        return cropped.extract_text() or ""
    except Exception:
        log.warning(
            "PDF table text exclusion failed",
            extra={
                "event": "pdf_table_text_exclusion_failed",
                "page": getattr(page, "page_number", None),
            },
        )
        return full_text


def _check_render_pixels(render_doc, index: int, scale: float, max_pixels: int) -> None:
    """`.render()` rasterizes the FULL page at `scale` regardless of whether the
    caller wants the whole page or just a crop of it -- so the size that matters
    is the page's, not the crop's. A page with a huge MediaBox renders to an
    enormous bitmap even on a document that otherwise passed max_pdf_pages."""
    width_pt, height_pt = render_doc[index].get_size()
    pixels = (width_pt * scale) * (height_pt * scale)
    if pixels > max_pixels:
        raise UnsafeContentError(
            f"page {index + 1} would render to ~{int(pixels):,} px "
            f"({width_pt * scale:.0f}x{height_pt * scale:.0f}), exceeds the "
            f"{max_pixels:,} px limit"
        )


def _render_page(
    render_doc, index: int, scale: float = RENDER_SCALE, max_pixels: int | None = None
):
    """Rasterize page `index` once. Raises UnsafeContentError (not caught here)
    if the render would exceed `max_pixels` -- callers decide whether that
    should abort the whole document (full-page OCR) or just skip this page's
    figures (crop captioning)."""
    if max_pixels is not None:
        _check_render_pixels(render_doc, index, scale, max_pixels)
    return render_doc[index].render(scale=scale).to_pil()


def _render_page_png(
    render_doc, index: int, scale: float = RENDER_SCALE, max_pixels: int | None = None
) -> bytes:
    pil = _render_page(render_doc, index, scale, max_pixels)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _crop_png(
    pil, im: dict, scale: float = RENDER_SCALE, excluded_bboxes: list[tuple] | None = None
):
    """Crop an already-rendered page bitmap to `im`'s bbox. Returns None on any
    failure (e.g. a malformed bbox) so one bad image doesn't abort the page."""
    try:
        box = (
            int(float(im["x0"]) * scale),
            int(float(im["top"]) * scale),
            int(float(im["x1"]) * scale),
            int(float(im["bottom"]) * scale),
        )
        crop = pil.crop(box)
        if excluded_bboxes:
            draw = ImageDraw.Draw(crop)
            for bbox in excluded_bboxes:
                bx0, btop, bx1, bbottom = map(float, bbox)
                masked = (
                    max(0, int(bx0 * scale) - box[0]),
                    max(0, int(btop * scale) - box[1]),
                    min(crop.width, int(math.ceil(bx1 * scale)) - box[0]),
                    min(crop.height, int(math.ceil(bbottom * scale)) - box[1]),
                )
                if masked[2] > masked[0] and masked[3] > masked[1]:
                    draw.rectangle(masked, fill="white")
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
