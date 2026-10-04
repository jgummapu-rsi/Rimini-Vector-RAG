import io
from types import SimpleNamespace

import pytest
from fpdf import FPDF
from PIL import Image

from app.ingest.pipeline.chunker import chunk_elements
from app.ingest.pipeline.layout import layout_profile
from app.ingest.pipeline.loaders.pdf import _native_word_regions, extract
from app.retrieval.rag.provenance import citation_provenance
from app.retrieval.rag.query import _quote_provenance


def test_native_quote_highlights_only_selected_words_after_chunking():
    document = FPDF()
    document.add_page()
    document.set_font("Helvetica", size=12)
    document.cell(0, 10, "Premium customers receive priority support during business hours.")
    cfg = SimpleNamespace(max_pdf_pages=5, max_image_pixels=4000000, layout_enabled=False)
    elements = extract(bytes(document.output()), "source.pdf", None, cfg)
    chunks = chunk_elements(elements)
    assert chunks
    provenance = citation_provenance(chunks[0].meta, "pdf")
    selected = _quote_provenance(provenance, "priority support")
    assert [region["text"] for region in selected["regions"]] == ["priority", "support"]
    assert all(region["precision"] == "word" for region in selected["regions"])
    assert selected["selection_status"] == "quote"


def test_layout_boundaries_do_not_split_same_baseline_words():
    class Page:
        def extract_words(self):
            return [
                {"text": "Left", "x0": 10, "x1": 30, "top": 10, "bottom": 20},
                {"text": "Right", "x0": 40, "x1": 70, "top": 10, "bottom": 20},
            ]

    regions = [{"id": 1, "bbox": [0, 0, 35, 30]}, {"id": 2, "bbox": [36, 0, 80, 30]}]
    output = _native_word_regions(Page(), [], "", regions)
    assert [line[0] for line in output] == ["Left Right"]


def test_enabled_layout_requires_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        layout_profile(
            SimpleNamespace(layout_enabled=True, layout_model_path=tmp_path / "missing.pt")
        )


def test_scanned_layout_keeps_completeness_and_region_precision():
    document = FPDF()
    document.add_page()
    image = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(image, format="PNG")
    image.seek(0)
    document.image(image, x=0, y=0, w=210, h=297)

    class Gateway:
        def layout(self, image):
            return [
                {"id": 0, "bbox": [0.1, 0.1, 0.9, 0.3], "label": "plain text", "confidence": 0.95}
            ]

        def vision(self, image, prompt, mime):
            return "Invoice total 100 USD"

    cfg = SimpleNamespace(max_pdf_pages=5, max_image_pixels=4000000, layout_enabled=True)
    elements = extract(bytes(document.output()), "scan.pdf", Gateway(), cfg)
    assert len(elements) == 1
    provenance = citation_provenance(elements[0].meta, "pdf")
    result = _quote_provenance(provenance, "Invoice total 100 USD")
    assert result["selection_status"] == "region"
    assert result["regions"][0]["height"] == pytest.approx(0.2)
