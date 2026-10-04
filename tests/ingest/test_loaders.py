"""Loader tests. Offline files must produce the right modality/extractor with NO
gateway call; the image loader must call vision and switch modality when the
model returns a markdown table."""

import io
import io as _io

import pdfplumber
import pypdfium2
import pytest
from fpdf import FPDF
from PIL import Image, ImageDraw

import app.ingest.pipeline.loaders.pdf as pdf_mod
from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.loaders import extract_document
from app.ingest.pipeline.loaders.pdf import (
    RENDER_SCALE,
    _check_render_pixels,
    _matching_region_spans,
    _matching_regions,
    _native_text_status,
)
from app.ingest.pipeline.prompts import FIGURE_EXTRACTION_PROMPT, STRICT_TRANSCRIBE_PROMPT
from app.ingest.pipeline.safety import UnsafeContentError
from app.shared.config import Settings
from app.shared.domain.models import Modality
from tests.conftest import FakeGateway, NoGateway

_CFG = Settings()


def test_csv_is_deterministic_table(files):
    els, summ = extract_document("data.csv", files["data.csv"], NoGateway(), _CFG)
    assert len(els) == 1
    assert els[0].modality == Modality.TABLE.value
    assert els[0].extractor == "csv_table"
    assert summ["by_modality"] == {"table": 1}


def test_txt_is_text(files):
    els, _ = extract_document("notes.txt", files["notes.txt"], NoGateway(), _CFG)
    assert all(e.modality == Modality.TEXT.value for e in els)


def test_docx_paragraphs_and_table_no_gateway(files):
    els, summ = extract_document("report.docx", files["report.docx"], NoGateway(), _CFG)
    mods = summ["by_modality"]
    assert mods.get("text", 0) >= 1 and mods.get("table", 0) == 1

    assert any(e.text.startswith("#") for e in els if e.modality == "text")


def test_docx_section_path_has_full_ancestor_chain(files):
    """DOCX must get the same full heading-path (not just nearest heading)
    markdown-sourced content gets -- _docx_bytes() nests Annual Report (H1) >
    Financials (H2), so a paragraph/table under Financials should carry both
    ancestors, not just the immediate one."""
    els, _ = extract_document("report.docx", files["report.docx"], NoGateway(), _CFG)
    financials_para = next(e for e in els if "Costs were controlled" in e.text)
    assert financials_para.meta.get("section_path") == "# Annual Report > ## Financials"

    tbl = next(e for e in els if e.modality == "table")
    assert tbl.meta.get("section_path") == "# Annual Report > ## Financials"
    assert tbl.meta.get("section") == "## Financials"


def test_xlsx_sheets_become_tables(files):
    els, summ = extract_document("finance.xlsx", files["finance.xlsx"], NoGateway(), _CFG)
    assert summ["by_extractor"].get("xlsx_table") == 2


def test_pdf_text_pages_no_gateway(files):
    els, summ = extract_document("doc.pdf", files["doc.pdf"], NoGateway(), _CFG)
    assert els and all(e.modality == Modality.TEXT.value for e in els)
    assert summ["by_extractor"].get("pdf_text", 0) >= 1


def test_pdf_native_text_quality_is_conservative_and_language_neutral():

    assert _native_text_status("Short title") == "sparse"
    assert _native_text_status("Quarterly revenue and operating margin improved.") == "usable"
    assert (
        _native_text_status(
            "\u6771\u4eac\u306e\u58f2\u4e0a\u5831\u544a\u306b\u306f\u56db\u534a\u671f\u306e\u8a73\u7d30\u306a\u6570\u5024\u304c\u8a18\u8f09\u3055\u308c\u3066\u3044\u307e\u3059"
        )
        == "usable"
    )
    assert _native_text_status("(cid:1)(cid:2)") == "corrupt"
    assert _native_text_status("Revenue " + "\ufffd" * 8 + " totals for Q4") == "corrupt"
    assert _native_text_status("(cid:12)(cid:13)(cid:14) extracted mapping") == "corrupt"


def test_pdf_corrupt_native_text_routes_to_full_page_vision(monkeypatch):

    class _Page:
        images = []

        def extract_text(self):
            return "Revenue " + "\ufffd" * 12 + " quarterly totals"

        def find_tables(self):
            return []

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        closed = False

        def __len__(self):
            return 1

        def close(self):
            self.closed = True

    render_doc = _RenderDoc()
    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: render_doc)
    monkeypatch.setattr(pdf_mod, "_render_page_png", lambda *_args, **_kwargs: b"png")

    gateway = FakeGateway("Quarterly totals were 125 units.")
    els, summary = extract_document("corrupt.pdf", b"pdf", gateway, _CFG)

    assert gateway.vision_calls == 1
    assert render_doc.closed
    assert len(els) == 1
    assert els[0].text == "Quarterly totals were 125 units."
    assert els[0].extractor == "vision"
    assert els[0].route_reason == "untrusted_text_layer"
    assert "\ufffd" not in els[0].text
    assert summary["by_reason"] == {"untrusted_text_layer": 1}


def test_corrupt_pdf_text_layer_aligns_vision_blocks_to_native_boxes():

    regions = [
        ("Bolanwar Gokul", (10, 20, 90, 35)),
        ("AI Engineer specializing in Generative AI", (10, 50, 190, 65)),
        ("Unrelated education", (10, 80, 100, 95)),
    ]

    assert (
        _matching_regions("Bolanwar Gokul\nAI Engineer specializing in Generative AI", regions)
        == regions[:2]
    )


def test_pdf_region_alignment_records_exact_offsets_and_wrapped_lines():

    text = "## Core Skills\n\n- **Retrieval & Search:** Vector Search, Embeddings, RAG Pipelines"
    regions = [
        ("CORE SKILLS", (10, 10, 80, 20)),
        ("Retrieval & Search: Vector Search, Embed-", (10, 30, 190, 40)),
        ("dings, RAG Pipelines", (20, 40, 100, 50)),
        ("Summary from another section", (10, 60, 150, 70)),
    ]

    spans = _matching_region_spans(
        text,
        regions,
        element=3,
        page=1,
        page_geometry={"page_width": 200, "page_height": 300},
    )

    assert [span["bbox"] for span in spans] == [region[1] for region in regions[:3]]
    assert all(span["start"] < span["end"] and span["precision"] == "exact_text" for span in spans)
    assert all(text[span["start"] : span["end"]] == span["text"] for span in spans)
    assert spans[1]["start"] > spans[0]["end"]


def test_pdf_region_alignment_handles_collapsed_spaces_and_hyphenated_wraps():

    text = "- **Retrieval & Search:** Enterprise Search, Vector Search, Embeddings, RAG Pipelines"
    regions = [
        ("Retrieval&Search: EnterpriseSearch,VectorSearch,Embed-", (10, 10, 190, 20)),
        ("dings,RAGPipelines", (20, 20, 100, 30)),
    ]

    spans = _matching_region_spans(text, regions, 0, 1, {})

    assert len(spans) == 2
    assert spans[0]["text"].startswith("Retrieval & Search")
    assert spans[0]["text"].endswith("Embed")
    assert spans[1]["text"] == "dings, RAG Pipelines"


def test_pdf_region_alignment_disambiguates_repeated_lines():

    text = "## Skills\nPython FastAPI\n\n## Experience\nPython FastAPI"
    regions = [
        ("Python FastAPI", (10, 20, 90, 30)),
        ("Python FastAPI", (10, 80, 90, 90)),
    ]

    spans = _matching_region_spans(text, regions, 0, 1, {})

    assert len(spans) == 2
    assert spans[0]["start"] < spans[1]["start"]


def test_pdf_usable_native_text_does_not_route_tiny_image_to_vision(monkeypatch):

    native = "Quarterly revenue and operating margin improved materially."

    class _Page:
        images = [{"x0": 0, "x1": 5, "top": 0, "bottom": 5}]

        def extract_text(self):
            return native

        def find_tables(self):
            return []

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())

    els, summary = extract_document("digital.pdf", b"pdf", NoGateway(), _CFG)

    evidence = [element for element in els if element.text]
    assert len(evidence) == 1
    assert evidence[0].text == native
    assert evidence[0].extractor == "pdf_text"
    assert evidence[0].route_reason == "text_layer"
    assert summary["omitted_regions"] == 1


def _pdf_with_small_logo(text: str) -> bytes:

    logo = Image.new("RGB", (240, 120), color="white")
    ImageDraw.Draw(logo).text((15, 45), "ACME", fill="black")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.text(10, 20, text)
    pdf.image(logo, x=10, y=30, w=20, h=10)
    return bytes(pdf.output())


def test_pdf_sparse_page_with_small_logo_uses_full_page_vision():
    gateway = FakeGateway("ACME")

    els, summary = extract_document("logo-cover.pdf", _pdf_with_small_logo("Q4"), gateway, _CFG)

    assert gateway.vision_calls == 1
    assert len(els) == 1
    assert els[0].text == "ACME"
    assert els[0].route_reason == "scanned_page"
    assert summary["by_reason"] == {"scanned_page": 1}


def test_pdf_text_page_keeps_readable_small_logo():
    gateway = FakeGateway("ACME")
    text = "Quarterly revenue and operating margin improved materially."

    els, summary = extract_document("logo-report.pdf", _pdf_with_small_logo(text), gateway, _CFG)

    assert gateway.vision_calls == 1
    assert [element.extractor for element in els] == ["pdf_text", "vision"]
    assert els[1].text == "ACME"
    assert els[1].modality == Modality.IMAGE.value
    assert els[1].route_reason == "figure_description"
    assert summary["by_reason"] == {"text_layer": 1, "figure_description": 1}


def _pdf_with_searchable_scan() -> bytes:

    image = Image.new("RGB", (1200, 1700), color="white")
    pdf = FPDF()
    pdf.add_page()
    pdf.image(image, x=0, y=0, w=210, h=297)
    pdf.set_font("Helvetica", size=12)
    pdf.text(20, 40, "Visible full page OCR text that already exists in the text layer.")
    return bytes(pdf.output())


def test_pdf_searchable_scan_does_not_duplicate_trusted_native_text():
    text = "Visible full page OCR text that already exists in the text layer."

    els, summary = extract_document(
        "searchable-scan.pdf", _pdf_with_searchable_scan(), NoGateway(), _CFG
    )

    assert len(els) == 1
    assert els[0].text == text
    assert els[0].extractor == "pdf_text"
    assert summary["by_reason"] == {"text_layer": 1}


def test_pdf_large_figure_without_native_text_overlap_is_still_transcribed():

    image = Image.new("RGB", (800, 800), color="white")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.text(10, 20, "Quarterly revenue and operating margin improved materially.")
    pdf.image(image, x=20, y=70, w=170, h=170)

    gateway = FakeGateway("Revenue chart labels")
    els, _ = extract_document("chart.pdf", bytes(pdf.output()), gateway, _CFG)

    assert gateway.vision_calls == 1
    assert [element.extractor for element in els] == ["pdf_text", "vision"]
    assert els[1].text == "Revenue chart labels"


def test_pdf_uses_distinct_prompts_for_scans_and_figures():

    scan_gateway = FakeGateway("ACME")
    extract_document("cover.pdf", _pdf_with_small_logo("Q4"), scan_gateway, _CFG)
    assert scan_gateway.vision_prompts == [STRICT_TRANSCRIBE_PROMPT]

    figure_gateway = FakeGateway("ACME")
    text = "Quarterly revenue and operating margin improved materially."
    extract_document("report.pdf", _pdf_with_small_logo(text), figure_gateway, _CFG)
    assert figure_gateway.vision_prompts == [FIGURE_EXTRACTION_PROMPT]


def test_pdf_corrupt_text_vision_failure_is_not_published_as_native(monkeypatch):

    class _Page:
        images = []

        def extract_text(self):
            return "(cid:10)(cid:11)(cid:12) broken font mapping"

        def find_tables(self):
            return []

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    class _FailingGateway:
        def vision(self, *_args, **_kwargs):
            raise RuntimeError("vision unavailable")

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())
    monkeypatch.setattr(pdf_mod, "_render_page_png", lambda *_args, **_kwargs: b"png")

    with pytest.raises(RuntimeError, match="vision unavailable"):
        extract_document("corrupt.pdf", b"pdf", _FailingGateway(), _CFG)


def _pdf_with_vector_graphics() -> bytes:

    pdf = FPDF()
    pdf.add_page()
    pdf.line(20, 20, 80, 80)
    pdf.line(80, 20, 20, 80)
    return bytes(pdf.output())


def _blank_pdf() -> bytes:

    pdf = FPDF()
    pdf.add_page()
    return bytes(pdf.output())


def test_pdf_vector_only_page_routes_to_full_page_vision():
    gateway = FakeGateway("Process flow from intake to approval.")

    els, summary = extract_document("vector.pdf", _pdf_with_vector_graphics(), gateway, _CFG)

    assert gateway.vision_calls == 1
    assert len(els) == 1
    assert els[0].text == "Process flow from intake to approval."
    assert els[0].extractor == "vision"
    assert els[0].route_reason == "visual_page_without_text_layer"
    assert els[0].meta["page"] == 1
    span = els[0].meta["source_spans"][0]
    assert {key: span[key] for key in ("element", "start", "end", "page")} == {
        "element": 0,
        "start": 0,
        "end": len(els[0].text),
        "page": 1,
    }
    assert span["page_width"] > 0 and span["page_height"] > 0
    assert summary["by_reason"] == {"visual_page_without_text_layer": 1}


def test_pdf_blank_page_does_not_call_vision():
    els, summary = extract_document("blank.pdf", _blank_pdf(), NoGateway(), _CFG)

    assert els == []
    assert summary == {
        "elements": 0,
        "by_extractor": {},
        "by_modality": {},
        "by_reason": {},
        "omitted_regions": 0,
        "extraction_status": "empty",
    }


def test_pdf_native_table_without_other_content_is_not_retranscribed(monkeypatch):

    class _CroppedPage:
        def extract_text(self):
            return ""

    class _Table:
        bbox = (0, 0, 100, 100)

        def extract(self):
            return [["Region", "Revenue"], ["APAC", "120"]]

    class _Page:
        images = []
        objects = {"line": [{"x0": 0}]}

        def extract_text(self):
            return "Region Revenue APAC 120"

        def find_tables(self):
            return [_Table()]

        def outside_bbox(self, _bbox):
            return _CroppedPage()

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())

    els, _ = extract_document("table.pdf", b"pdf", NoGateway(), _CFG)

    assert len(els) == 1
    assert els[0].extractor == "pdf_table"
    assert "APAC" in els[0].text
    assert els[0].meta["table_index"] == 0


def test_pdf_mixed_content_is_emitted_in_page_order():

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.text(10, 20, "INTRO_SENTINEL before the revenue table.")
    for y in (40, 50, 60):
        pdf.line(10, y, 150, y)
    for x in (10, 80, 150):
        pdf.line(x, 40, x, 60)
    for x, y, value in (
        (12, 47, "Region"),
        (82, 47, "Revenue"),
        (12, 57, "APAC"),
        (82, 57, "120"),
    ):
        pdf.text(x, y, value)
    pdf.text(10, 80, "CONCLUSION_SENTINEL after the table.")

    els, _ = extract_document("ordered.pdf", bytes(pdf.output()), NoGateway(), _CFG)

    assert [element.modality for element in els] == ["text", "table", "text"]
    assert "INTRO_SENTINEL" in els[0].text
    assert "APAC" in els[1].text
    assert "CONCLUSION_SENTINEL" in els[2].text
    assert [element.order for element in els] == [0, 1, 2]


def test_pdf_figure_is_emitted_between_surrounding_text(monkeypatch):

    words = [
        {"text": "Before", "x0": 10, "x1": 40, "top": 10, "bottom": 20},
        {"text": "After", "x0": 10, "x1": 35, "top": 90, "bottom": 100},
    ]

    class _Page:
        width = 120
        height = 120
        chars = []
        images = [{"x0": 10, "x1": 80, "top": 40, "bottom": 80}]
        objects = {}

        def extract_text(self):
            return "Before\nAfter with enough native text to stay usable."

        def extract_words(self):
            return words

        def find_tables(self):
            return []

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())
    monkeypatch.setattr(
        pdf_mod,
        "_render_page",
        lambda *_args, **_kwargs: Image.new("RGB", (400, 400), "white"),
    )

    els, _ = extract_document("figure-order.pdf", b"pdf", FakeGateway("Figure"), _CFG)

    assert [element.text for element in els] == ["Before", "Figure", "After"]
    assert [element.order for element in els] == [0, 1, 2]


def test_pdf_table_failure_is_logged_and_later_tables_are_kept(monkeypatch, caplog):

    class _BrokenTable:
        bbox = (0, 0, 20, 20)

        def extract(self):
            raise ValueError("malformed cells")

    class _GoodTable:
        bbox = (30, 30, 80, 80)

        def extract(self):
            return [["Region", "Revenue"], ["APAC", "120"]]

    class _Page:
        page_number = 1
        images = []
        objects = {}

        def extract_text(self):
            return "Body text remains available after a malformed table."

        def find_tables(self):
            return [_BrokenTable(), _GoodTable()]

        def outside_bbox(self, _bbox):
            return self

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())

    with caplog.at_level("WARNING", logger="pipeline.pdf"):
        els, _ = extract_document("tables.pdf", b"pdf", NoGateway(), _CFG)

    assert any(element.meta.get("table_index") == 1 for element in els)
    assert any(record.event == "pdf_table_extraction_failed" for record in caplog.records)


def test_pdf_table_detection_failure_is_logged_and_text_falls_back(monkeypatch, caplog):

    class _Page:
        page_number = 1
        images = []
        objects = {}

        def extract_text(self):
            return "Readable native text survives table detector failure."

        def find_tables(self):
            raise RuntimeError("detector failed")

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())

    with caplog.at_level("WARNING", logger="pipeline.pdf"):
        els, _ = extract_document("tables.pdf", b"pdf", NoGateway(), _CFG)

    assert [element.text for element in els if element.text] == [
        "Readable native text survives table detector failure."
    ]
    assert any(element.meta.get("omission") == "table_detection_failed" for element in els)
    assert any(record.event == "pdf_table_detection_failed" for record in caplog.records)


def test_pdf_image_fully_covered_by_native_table_is_not_transcribed(monkeypatch):

    class _Table:
        bbox = (10, 10, 110, 110)

        def extract(self):
            return [["Region", "Revenue"], ["APAC", "120"]]

    class _Page:
        images = [{"x0": 10, "x1": 110, "top": 10, "bottom": 110}]
        objects = {}

        def extract_text(self):
            return "Region Revenue APAC 120"

        def find_tables(self):
            return [_Table()]

        def outside_bbox(self, _bbox):
            return type("Cropped", (), {"extract_text": lambda self: ""})()

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())

    els, _ = extract_document("table-image.pdf", b"pdf", NoGateway(), _CFG)

    assert len(els) == 1
    assert els[0].extractor == "pdf_table"


def test_pdf_native_table_region_is_masked_from_overlapping_figure(monkeypatch):

    class _Table:
        bbox = (0, 0, 20, 20)

        def extract(self):
            return [["Region", "Revenue"], ["APAC", "120"]]

    class _Page:
        images = [{"x0": 0, "x1": 100, "top": 0, "bottom": 100}]
        objects = {}

        def extract_text(self):
            return "Region Revenue APAC 120"

        def find_tables(self):
            return [_Table()]

        def outside_bbox(self, _bbox):
            return type("Cropped", (), {"extract_text": lambda self: ""})()

    class _Pdf:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _RenderDoc:
        def __len__(self):
            return 1

        def close(self):
            pass

    class _Gateway:
        image = None

        def vision(self, image_bytes, *_args, **_kwargs):
            self.image = Image.open(io.BytesIO(image_bytes)).copy()
            return "Chart labels"

    gateway = _Gateway()
    monkeypatch.setattr(pdfplumber, "open", lambda *_args, **_kwargs: _Pdf())
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _RenderDoc())
    monkeypatch.setattr(
        pdf_mod,
        "_render_page",
        lambda *_args, **_kwargs: Image.new("RGB", (300, 300), "black"),
    )

    els, _ = extract_document("mixed.pdf", b"pdf", gateway, _CFG)

    assert [element.extractor for element in els] == ["pdf_table", "vision"]
    assert gateway.image.getpixel((10, 10)) == (255, 255, 255)
    assert gateway.image.getpixel((200, 200)) == (0, 0, 0)


def test_pdf_vector_only_page_propagates_vision_failure():
    class _FailingGateway:
        def vision(self, *_args, **_kwargs):
            raise RuntimeError("vision unavailable")

    with pytest.raises(RuntimeError, match="vision unavailable"):
        extract_document("vector.pdf", _pdf_with_vector_graphics(), _FailingGateway(), _CFG)


def _pdf_with_two_figures() -> bytes:
    """A single page with body text (so it's routed as a text page, not a
    scanned page) plus two readable embedded images -- both crops must come
    from the SAME page render."""

    img = Image.new("RGB", (150, 150), color="red")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "This page has body text alongside two embedded figures for captioning.")
    pdf.image(img, x=10, y=60, w=60, h=60)
    pdf.image(img, x=100, y=60, w=60, h=60)
    return bytes(pdf.output())


def test_pdf_page_bitmap_is_rendered_once_and_reused_for_every_crop(monkeypatch):
    """Finding 1.10a: `.render()` rasterizes the whole page -- re-running it
    once per embedded image is wasted work. The page bitmap must be rendered
    once and reused for every qualifying crop on that page."""

    real_render_page = pdf_mod._render_page
    calls = []

    def _counting_render_page(*args, **kwargs):
        calls.append(1)
        return real_render_page(*args, **kwargs)

    monkeypatch.setattr(pdf_mod, "_render_page", _counting_render_page)

    gw = FakeGateway("A red square.")
    els, summ = extract_document("figs.pdf", _pdf_with_two_figures(), gw, _CFG)

    assert gw.vision_calls == 2
    assert len(calls) == 1


def test_pdf_page_render_over_pixel_limit_raises():
    """Finding 1.3: `_check_render_pixels` bounds the PAGE render, not just the
    crop -- a huge page must be rejected even though max_pdf_pages passed."""

    class _FakePage:
        def get_size(self):
            return (10000.0, 10000.0)

    class _FakeDoc(list):
        pass

    doc = _FakeDoc([_FakePage()])
    with pytest.raises(UnsafeContentError):
        _check_render_pixels(doc, 0, RENDER_SCALE, max_pixels=1_000_000)

    _check_render_pixels(doc, 0, RENDER_SCALE, max_pixels=10**12)


def test_image_routes_to_vision_as_image():
    gw = FakeGateway("A photo of a warehouse with text SHIP NOW.")
    els, _ = extract_document("pic.png", _tiny_png(), gw, _CFG)
    assert gw.vision_calls == 1
    assert els[0].modality == Modality.IMAGE.value
    assert els[0].extractor == "vision"


def test_image_table_returns_markdown_modality_table():

    gw = FakeGateway("| a | b |\n| --- | --- |\n| 1 | 2 |")
    els, _ = extract_document("grid.jpg", _tiny_png(), gw, _CFG)
    assert els[0].modality == Modality.TABLE.value
    assert els[0].extractor == "vision_table"


def _tiny_png() -> bytes:
    """A real (tiny) 1x1 PNG -- the image loader now opens it with PIL to check
    dimensions before sending it to vision, so a fake non-image byte string is
    no longer a valid fixture."""

    buf = _io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    return buf.getvalue()


def test_markdown_tables_become_table_elements_with_caption():
    md = (
        b"# Title\n\n"
        b"Some intro prose here.\n\n"
        b"## 1. Core Inference\n\n"
        b"| Endpoint | Use |\n|---|---|\n"
        b"| POST /v1/chat | chat |\n| POST /v1/ocr | ocr |\n\n"
        b"```bash\ncurl http://x\n```\n"
    )
    els, summ = extract_document("api.md", md, NoGateway(), _CFG)
    mods = summ["by_modality"]
    assert mods.get("table", 0) == 1
    tbl = next(e for e in els if e.modality == "table")
    assert tbl.extractor == "markdown_table"
    assert tbl.meta.get("section") == "## 1. Core Inference"
    assert "| Endpoint | Use |" in tbl.text

    assert any(e.route_reason == "code_block" for e in els)


def test_markdown_big_table_row_splits_repeat_caption_and_header():
    header = "## Endpoints\n| Endpoint | Use |\n| --- | --- |"
    rows = "\n".join(f"| POST /v1/x{i} | does thing number {i} here |" for i in range(60))
    el = Element(
        header + "\n" + rows,
        Modality.TABLE.value,
        "markdown_table",
        "structured_table",
        0,
        {"section": "## Endpoints"},
    )
    spec = ChunkSpec(target_tokens=80, overlap_tokens=10, max_tokens=120, min_tokens=8)
    recs = chunk_elements([el], spec)
    assert len(recs) > 1
    for r in recs:
        head = r.text.splitlines()[:3]
        assert head[0] == "## Endpoints"
        assert head[1] == "| Endpoint | Use |"
        assert r.token_count <= spec.max_tokens
