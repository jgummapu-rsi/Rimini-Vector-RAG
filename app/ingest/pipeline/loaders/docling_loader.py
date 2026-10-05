"""CPU PDF/image conversion into the existing ordered, source-aware Elements.

Heavy imports stay inside the isolated parser process. Structured Docling items
are adapted directly; a document-wide Markdown round trip would lose geometry,
table boundaries, heading hierarchy, and item identity.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.docling_config import DOCLING_REVISION, IMAGE_EXTENSIONS
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.outcomes import omission
from app.ingest.pipeline.prompts import FIGURE_EXTRACTION_PROMPT
from app.ingest.pipeline.safety import UnsafeContentError


def _preflight(data: bytes, filename: str, cfg) -> None:
    if Path(filename).suffix.lower() == ".pdf":
        with pdfium.PdfDocument(data) as pdf:
            if len(pdf) > cfg.max_pdf_pages:
                raise UnsafeContentError("PDF exceeds page limit")
            for page in pdf:
                try:
                    width, height = page.get_size()
                    # OCR uses 3x rendering; table/figure rendering uses 2x.
                    if math.ceil(width * 3) * math.ceil(height * 3) > cfg.max_image_pixels:
                        raise UnsafeContentError("PDF OCR render exceeds pixel limit")
                finally:
                    page.close()
    else:
        with Image.open(io.BytesIO(data)) as image:
            frames = getattr(image, "n_frames", 1)
            if frames > cfg.max_pdf_pages:
                raise UnsafeContentError("Image exceeds frame limit")
            for frame in range(frames):
                image.seek(frame)
                dpi = image.info.get("dpi")
                if dpi in (None, (1, 1)):
                    dpi = (72, 72)
                if not isinstance(dpi, tuple) or len(dpi) != 2:
                    raise UnsafeContentError("Invalid image DPI")
                dx, dy = map(float, dpi)
                if not all(math.isfinite(d) and d > 0 for d in (dx, dy)):
                    raise UnsafeContentError("Invalid image DPI")
                pixels = math.ceil(image.width * 216 / dx) * math.ceil(image.height * 216 / dy)
                if max(pixels, image.width * image.height) > cfg.max_image_pixels:
                    raise UnsafeContentError("Image OCR render exceeds pixel limit")


def _converter(cfg):
    from docling.datamodel.accelerator_options import (  # noqa: PLC0415
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.object_detection_engine_options import (  # noqa: PLC0415
        TransformersObjectDetectionEngineOptions,
    )
    from docling.datamodel.pipeline_options import (  # noqa: PLC0415
        HeadingHierarchyOptions,
        LayoutObjectDetectionOptions,
        OcrMode,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import (  # noqa: PLC0415
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    options = PdfPipelineOptions(
        artifacts_path=Path(cfg.docling_artifacts_path),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=cfg.docling_num_threads
        ),
        document_timeout=cfg.parse_timeout_seconds,
        enable_remote_services=False,
        allow_external_plugins=False,
        do_ocr=True,
        ocr_options=RapidOcrOptions(
            backend="onnxruntime",
            lang=[cfg.docling_ocr_language],
            model_size=cfg.docling_ocr_model_size,
            mode=OcrMode.FULL_PAGE if cfg.docling_force_full_page_ocr else OcrMode.DEFAULT,
            scale=3.0,
        ),
        do_table_structure=True,
        layout_options=LayoutObjectDetectionOptions(
            engine_options=TransformersObjectDetectionEngineOptions(
                compile_model=False, torch_dtype="float32"
            )
        ),
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode(cfg.docling_table_mode), do_cell_matching=True
        ),
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_description=False,
        do_picture_classification=False,
        generate_page_images=False,
        generate_picture_images=cfg.docling_picture_description,
        images_scale=2.0,
        ocr_batch_size=cfg.docling_batch_size,
        layout_batch_size=cfg.docling_batch_size,
        table_batch_size=cfg.docling_batch_size,
        queue_max_size=max(2, cfg.docling_batch_size * 2),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        },
    )


def _metadata(item, document, text: str, index: int, *, image_source: bool) -> dict:
    meta = {
        "docling_ref": item.self_ref,
        "docling_revision": DOCLING_REVISION,
        "layout_label": item.label.value,
        "layout_id": item.self_ref,
        "source_element": index,
        "precision": "element_region",
    }
    spans = []
    for prov in item.prov:
        page = document.pages.get(prov.page_no)
        if page is None:
            continue
        box = prov.bbox.to_top_left_origin(page_height=page.size.height)
        spans.append(
            {
                "element": index,
                "start": 0,
                "end": len(text),
                "text": text,
                "page": prov.page_no,
                "bbox": [box.l, box.t, box.r, box.b],
                "page_width": page.size.width,
                "page_height": page.size.height,
                "page_box": [0, 0, page.size.width, page.size.height],
                "precision": "element_region",
                "layout_id": item.self_ref,
                "layout_label": item.label.value,
                **({"frame": prov.page_no} if image_source else {}),
            }
        )
    if spans:
        meta.update(
            {
                key: value
                for key, value in spans[0].items()
                if key not in {"element", "start", "end", "text"}
            }
        )
        meta["pages"] = sorted({span["page"] for span in spans})
        meta["source_spans"] = spans
    return meta


def extract(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    _preflight(data, filename, cfg)
    from docling.datamodel.base_models import ConversionStatus, DocumentStream  # noqa: PLC0415
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem  # noqa: PLC0415

    result = _converter(cfg).convert(
        DocumentStream(name=filename, stream=io.BytesIO(data)),
        raises_on_error=False,
        max_num_pages=cfg.max_pdf_pages,
        max_file_size=len(data),
    )
    # Never publish a partial conversion as successful evidence.
    if result.status != ConversionStatus.SUCCESS:
        raise UnsafeContentError(f"Docling conversion was not complete: {result.status.value}")
    document = result.document
    image_source = Path(filename).suffix.lower() in IMAGE_EXTENSIONS
    elements: list[Element] = []
    headings: list[tuple[int, str]] = []
    extracted_bytes = 0
    for item, _ in document.iterate_items(with_groups=False, traverse_pictures=False):
        label = item.label.value
        # Rich table cell text is already serialized by TableItem. Captions and
        # footnotes remain separate evidence, and pictures still need vision.
        if not isinstance(item, (TableItem, PictureItem)) and label not in {"caption", "footnote"}:
            parent = item.parent
            inside_table = False
            while parent is not None:
                ancestor = parent.resolve(document)
                if isinstance(ancestor, TableItem):
                    inside_table = True
                    break
                parent = ancestor.parent
            if inside_table:
                continue
        text = getattr(item, "text", "") or ""
        modality, reason = "text", "docling_reading_order"
        if isinstance(item, SectionHeaderItem) or label == "title":
            level = item.level if isinstance(item, SectionHeaderItem) else 0
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, text))
        if isinstance(item, TableItem):
            if item.data.num_rows > cfg.max_table_rows:
                raise UnsafeContentError("Docling table exceeds row limit")
            if item.data.num_rows * item.data.num_cols > cfg.max_workbook_cells:
                raise UnsafeContentError("Docling table exceeds cell limit")
            text = item.export_to_markdown(doc=document)
            modality, reason = "table", "docling_table_structure"
        elif isinstance(item, PictureItem):
            meta = _metadata(item, document, "", len(elements), image_source=image_source)
            if not cfg.docling_picture_description:
                elements.append(
                    omission("docling", "picture_description_disabled", len(elements), **meta)
                )
                continue
            picture = item.get_image(document)
            if picture is None:
                elements.append(
                    omission("docling", "picture_image_unavailable", len(elements), **meta)
                )
                continue
            if picture.width * picture.height > cfg.max_image_pixels:
                raise UnsafeContentError("Docling picture exceeds pixel limit")
            buffer = io.BytesIO()
            picture.convert("RGB").save(buffer, format="PNG")
            text = gateway.vision(buffer.getvalue(), FIGURE_EXTRACTION_PROMPT, "image/png").strip()
            if not text:
                elements.append(
                    omission("docling", "picture_description_empty", len(elements), **meta)
                )
                continue
            # Use the existing splitter so tables transcribed from diagrams remain
            # first-class tables. Rebuild source spans for each resulting block.
            blocks = split_blocks(
                text,
                text_extractor="vision",
                table_extractor="vision",
                text_reason="docling_picture",
                table_reason="docling_picture_table",
                text_modality="image",
            )
            for block in blocks:
                block.order = len(elements)
                block.meta.update(
                    _metadata(item, document, block.text, block.order, image_source=image_source)
                )
                if headings:
                    block.meta.update(
                        section=headings[-1][1], section_path=" > ".join(h for _, h in headings)
                    )
                elements.append(block)
            extracted_bytes += len(text.encode())
            if extracted_bytes > cfg.max_extracted_bytes:
                raise UnsafeContentError("Docling extraction exceeds byte limit")
            continue
        if not text.strip():
            continue
        meta = _metadata(item, document, text, len(elements), image_source=image_source)
        if headings:
            meta.update(section=headings[-1][1], section_path=" > ".join(h for _, h in headings))
        if label == "code":
            meta["block_type"] = "code"
            reason = "code_block"
        if modality == "table":
            meta["table_index"] = int(item.self_ref.rsplit("/", 1)[-1])
        elements.append(Element(text, modality, "docling", reason, len(elements), meta))
        extracted_bytes += len(text.encode())
        if extracted_bytes > cfg.max_extracted_bytes:
            raise UnsafeContentError("Docling extraction exceeds byte limit")
    if not elements and data:
        raise UnsafeContentError("Docling produced no evidence from a nonempty source")
    return elements
