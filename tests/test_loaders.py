"""Loader tests. Offline files must produce the right modality/extractor with NO
gateway call; the image loader must call vision and switch modality when the
model returns a markdown table."""
from app.domain.models import Modality
from app.pipeline.loaders import extract_document
from tests.conftest import FakeGateway, NoGateway


def test_csv_is_deterministic_table(files):
    els, summ = extract_document("data.csv", files["data.csv"], NoGateway())
    assert len(els) == 1
    assert els[0].modality == Modality.TABLE.value
    assert els[0].extractor == "csv_table"
    assert summ["by_modality"] == {"table": 1}


def test_txt_is_text(files):
    els, _ = extract_document("notes.txt", files["notes.txt"], NoGateway())
    assert all(e.modality == Modality.TEXT.value for e in els)


def test_docx_paragraphs_and_table_no_gateway(files):
    els, summ = extract_document("report.docx", files["report.docx"], NoGateway())
    mods = summ["by_modality"]
    assert mods.get("text", 0) >= 1 and mods.get("table", 0) == 1
    # heading preserved as markdown
    assert any(e.text.startswith("#") for e in els if e.modality == "text")


def test_docx_section_path_has_full_ancestor_chain(files):
    """DOCX must get the same full heading-path (not just nearest heading)
    markdown-sourced content gets -- _docx_bytes() nests Annual Report (H1) >
    Financials (H2), so a paragraph/table under Financials should carry both
    ancestors, not just the immediate one."""
    els, _ = extract_document("report.docx", files["report.docx"], NoGateway())
    financials_para = next(e for e in els if "Costs were controlled" in e.text)
    assert financials_para.meta.get("section_path") == "# Annual Report > ## Financials"

    tbl = next(e for e in els if e.modality == "table")
    assert tbl.meta.get("section_path") == "# Annual Report > ## Financials"
    assert tbl.meta.get("section") == "## Financials"


def test_xlsx_sheets_become_tables(files):
    els, summ = extract_document("finance.xlsx", files["finance.xlsx"], NoGateway())
    assert summ["by_extractor"].get("xlsx_table") == 2  # two sheets


def test_pdf_text_pages_no_gateway(files):
    els, summ = extract_document("doc.pdf", files["doc.pdf"], NoGateway())
    assert els and all(e.modality == Modality.TEXT.value for e in els)
    assert summ["by_extractor"].get("pdf_text", 0) >= 1


def test_image_routes_to_vision_as_image():
    gw = FakeGateway("A photo of a warehouse with text SHIP NOW.")
    els, _ = extract_document("pic.png", b"\x89PNG_fake", gw)
    assert gw.vision_calls == 1
    assert els[0].modality == Modality.IMAGE.value
    assert els[0].extractor == "vision"


def test_image_table_returns_markdown_modality_table():
    # LLM returns a markdown table -> modality flips to table (the "LLM only for
    # image tables" rule).
    gw = FakeGateway("| a | b |\n| --- | --- |\n| 1 | 2 |")
    els, _ = extract_document("grid.jpg", b"fakejpg", gw)
    assert els[0].modality == Modality.TABLE.value
    assert els[0].extractor == "vision_table"


def test_markdown_tables_become_table_elements_with_caption():
    md = (
        "# Title\n\n"
        "Some intro prose here.\n\n"
        "## 1. Core Inference\n\n"
        "| Endpoint | Use |\n|---|---|\n"
        "| POST /v1/chat | chat |\n| POST /v1/ocr | ocr |\n\n"
        "```bash\ncurl http://x\n```\n"
    ).encode()
    from tests.conftest import NoGateway
    els, summ = extract_document("api.md", md, NoGateway())
    mods = summ["by_modality"]
    assert mods.get("table", 0) == 1
    tbl = next(e for e in els if e.modality == "table")
    assert tbl.extractor == "markdown_table"
    assert tbl.meta.get("section") == "## 1. Core Inference"
    assert "| Endpoint | Use |" in tbl.text
    # code block kept atomic as text
    assert any(e.route_reason == "code_block" for e in els)


def test_markdown_big_table_row_splits_repeat_caption_and_header():
    from app.pipeline.chunker import ChunkSpec, chunk_elements
    from app.domain.models import Modality
    from app.pipeline.elements import Element
    header = "## Endpoints\n| Endpoint | Use |\n| --- | --- |"
    rows = "\n".join(f"| POST /v1/x{i} | does thing number {i} here |" for i in range(60))
    el = Element(header + "\n" + rows, Modality.TABLE.value, "markdown_table",
                 "structured_table", 0, {"section": "## Endpoints"})
    spec = ChunkSpec(target_tokens=80, overlap_tokens=10, max_tokens=120, min_tokens=8)
    recs = chunk_elements([el], spec)
    assert len(recs) > 1
    for r in recs:
        head = r.text.splitlines()[:3]
        assert head[0] == "## Endpoints"
        assert head[1] == "| Endpoint | Use |"
        assert r.token_count <= spec.max_tokens
