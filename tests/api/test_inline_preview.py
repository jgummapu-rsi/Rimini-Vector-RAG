import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.api import ingest_routes
from tests.conftest import _docx_bytes, _xlsx_bytes


def source(monkeypatch, filename, data):
    item = SimpleNamespace(filename=filename, blob_path="test", version=1)
    monkeypatch.setattr(ingest_routes, "_source_version", lambda *args: (None, item))
    return SimpleNamespace(
        blob=SimpleNamespace(get=lambda path: data),
        settings=SimpleNamespace(
            max_image_pixels=100000, max_table_rows=1000, max_workbook_cells=10000
        ),
    )


@pytest.mark.parametrize(
    "filename,data,expected",
    [
        ("notes.md", b"# Heading\n\nEvidence", "Evidence"),
        ("notes.txt", b"Source text", "Source text"),
        ("notes.html", b"<script>alert(1)</script><p>Evidence</p>", "<script>"),
        ("notes.rtf", b"{\\rtf1\\ansi Source evidence}", "Source evidence"),
    ],
)
def test_exact_version_preview_preserves_text_as_data(monkeypatch, filename, data, expected):
    container = source(monkeypatch, filename, data)
    result = ingest_routes.preview_document_version("doc", 1, None, container)
    assert expected in result["text"]
    assert result["version"] == 1


def test_docx_preview_retains_body_anchors(monkeypatch):
    container = source(monkeypatch, "document.docx", _docx_bytes())
    result = ingest_routes.preview_document_version("doc", 1, None, container)
    assert result["blocks"][0]["locator"] == {"body_index": 1}
    assert "Annual Report" in result["blocks"][0]["text"]
    assert any("APAC" in block["text"] for block in result["blocks"])


def test_spreadsheet_preview_retains_sheet_and_rows(monkeypatch):
    container = source(monkeypatch, "book.xlsx", _xlsx_bytes())
    result = ingest_routes.preview_document_version("doc", 1, None, container)
    assert result["blocks"][1]["locator"] == {"sheet": "Sales", "row": 2}
    assert "APAC" in result["blocks"][1]["text"]


def test_csv_quoted_multiline_cell_is_one_source_row(monkeypatch):
    container = source(monkeypatch, "data.csv", b'id,value\n1,"first\nsecond"\n')
    result = ingest_routes.preview_document_version("doc", 1, None, container)
    assert len(result["blocks"]) == 2
    assert result["blocks"][1] == {"text": "1\tfirst\nsecond", "locator": {"row": 2}}


def test_tiff_renders_requested_frame(monkeypatch):
    buffer = io.BytesIO()
    with Image.new("RGB", (20, 20), "red") as first, Image.new("RGB", (20, 20), "blue") as second:
        first.save(buffer, "TIFF", save_all=True, append_images=[second])
    container = source(monkeypatch, "scan.tiff", buffer.getvalue())
    result = ingest_routes.render_image_frame("doc", 1, 2, None, container)
    with Image.open(io.BytesIO(result.body)) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    with pytest.raises(ingest_routes.HTTPException) as error:
        ingest_routes.render_image_frame("doc", 1, 3, None, container)
    assert error.value.status_code == 404
