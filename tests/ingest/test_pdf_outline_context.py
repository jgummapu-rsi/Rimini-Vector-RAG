from types import SimpleNamespace

from fpdf import FPDF

from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.loaders import extract_document
from app.ingest.pipeline.loaders.pdf import _outline_sections
from app.shared.config import Settings
from tests.conftest import NoGateway


def test_bookmarked_bundle_preserves_subject_across_pages_and_switches_at_boundary():
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for title, text in [
        ("Product Alpha", "Product Alpha maintenance manual."),
        (None, "Replace filter every 90 days. Use part FLT-00123."),
        ("Product Beta", "Product Beta uses part FLT-00456 every 30 days."),
    ]:
        pdf.add_page()
        if title:
            pdf.start_section(title)
        pdf.multi_cell(0, 8, text)
    elements, _ = extract_document(
        "bundle.pdf", bytes(pdf.output()), NoGateway(), Settings(_env_file=None)
    )
    chunks = chunk_elements(
        elements, ChunkSpec.auto(8191), count=lambda text: len(text.split()), embed_max=8191
    )
    alpha = next(chunk for chunk in chunks if "90 days" in chunk.text)
    beta = next(chunk for chunk in chunks if "30 days" in chunk.text)
    assert alpha.text.startswith("Product Alpha\n\n")
    assert "Product Beta" not in alpha.text
    assert alpha.meta["section_start_page"] == 1
    assert alpha.meta["section_end_page"] == 2
    assert beta.text.startswith("Product Beta\n\n")
    assert all(span.get("text") for chunk in chunks for span in chunk.meta["source_spans"])


def test_ambiguous_same_page_siblings_do_not_assign_wrong_subject():

    class Document:
        def __len__(self):
            return 3

        def get_toc(self):
            for title, page, level in [
                ("Manual", 0, 0),
                ("Alpha", 0, 1),
                ("Beta", 1, 1),
                ("Gamma", 1, 1),
            ]:
                yield SimpleNamespace(
                    level=level,
                    get_title=lambda t=title: t,
                    get_dest=lambda p=page: SimpleNamespace(get_index=lambda: p),
                )

    sections = _outline_sections(Document())
    assert sections[1]["section_path"] == "Manual > Alpha"
    assert 2 not in sections and 3 not in sections
