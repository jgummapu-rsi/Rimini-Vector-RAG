import copy
import io

from docx import Document

from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.loaders import extract_document
from app.shared.config import Settings


def line(index, *, page=1, x=20, y=None):
    text = f"Evidence line {index:03d} describes a unique requirement."
    meta = {
        "page": page,
        "bbox": [
            x,
            y if y is not None else index * 12,
            x + 200,
            (y if y is not None else index * 12) + 10,
        ],
        "page_width": 600,
        "page_height": 800,
        "source_element": index,
    }
    meta["source_spans"] = [
        {
            "element": index,
            "start": 0,
            "end": len(text),
            "text": text,
            **{k: meta[k] for k in ("page", "bbox", "page_width", "page_height")},
        }
    ]
    return Element(text, "text", "pdf_text", "text_layer", index, meta)


def test_native_lines_pack_into_context_without_losing_source_regions():
    elements = [line(i) for i in range(30)]
    spec = ChunkSpec(target_tokens=60, overlap_tokens=10, max_tokens=80, min_tokens=8)
    chunks = chunk_elements(elements, spec, count=lambda text: len(text.split()), embed_max=80)
    assert 1 < len(chunks) < 10
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert all(element.text in "\n".join(chunk.text for chunk in chunks) for element in elements)
    for chunk in chunks:
        assert "bbox" not in chunk.meta
        for span in chunk.meta["source_spans"]:
            original = elements[span["element"]]
            assert span["bbox"] == original.meta["bbox"]
            assert original.text[span["element_start"] : span["element_end"]] == span["text"]
            assert span["text"] in chunk.text


def test_pages_columns_large_gaps_and_tables_separate_context():
    elements = [
        line(0),
        line(1),
        line(2, x=330, y=0),
        line(3, x=330, y=12),
        line(4, page=2, y=0),
        line(5, page=2, y=12),
        line(6, page=2, y=200),
        line(7, page=2, y=212),
    ]
    chunks = chunk_elements(elements, count=lambda text: len(text.split()))
    assert len(chunks) == 4
    assert all(len(chunk.meta["pages"]) == 1 for chunk in chunks)
    table = Element("| Key | Value |\n| --- | --- |\n| A | 10 |", "table", "pdf_table", "table", 10)
    separated = chunk_elements([line(0), table, line(1)])
    assert [chunk.modality for chunk in separated] == ["text", "table", "text"]


def test_section_change_is_not_merged_even_when_lines_are_adjacent():
    elements = [line(i) for i in range(4)]
    for i, element in enumerate(elements):
        element.meta["section_path"] = "First" if i < 2 else "Second"
    chunks = chunk_elements(elements)
    assert len(chunks) == 2
    assert [chunk.meta["section_path"] for chunk in chunks] == ["First", "Second"]


def test_packing_does_not_mutate_cached_elements():
    elements = [line(0), line(1)]
    before = copy.deepcopy(elements)
    chunk_elements(elements)
    assert elements == before


def test_layout_detections_and_indentation_do_not_fragment_prose():
    elements = [line(i, x=20 if i % 2 else 60) for i in range(20)]
    for i, element in enumerate(elements):
        element.meta.update(layout_id=i, layout_label="plain text", layout_confidence=0.9)
        element.meta["source_spans"][0]["layout_id"] = i

    chunks = chunk_elements(
        elements,
        ChunkSpec(target_tokens=60, overlap_tokens=0, max_tokens=80),
        count=lambda text: len(text.split()),
        embed_max=80,
    )

    assert len(chunks) == 3
    assert all("layout_id" not in chunk.meta and "bbox" not in chunk.meta for chunk in chunks)
    for chunk in chunks:
        for span in chunk.meta["source_spans"]:
            original = elements[span["element"]]
            assert span["layout_id"] == original.meta["layout_id"]
            assert span["bbox"] == original.meta["bbox"]
            assert original.text[span["element_start"] : span["element_end"]] == span["text"]
            assert span["text"] in chunk.text


def test_centered_heading_stays_with_indented_body():
    heading = line(0, x=200, y=0)
    heading.text = "PROFESSIONAL SUMMARY"
    heading.meta.pop("source_spans")
    body = line(1, x=20, y=24)
    chunks = chunk_elements([heading, body])
    assert len(chunks) == 1
    assert heading.text in chunks[0].text and body.text in chunks[0].text


def test_loader_source_ids_do_not_prevent_paragraph_packing():

    document = Document()
    for i in range(12):
        document.add_paragraph(f"Paragraph {i} describes a distinct requirement.")
    output = io.BytesIO()
    document.save(output)
    elements, _ = extract_document("requirements.docx", output.getvalue(), None, Settings())
    assert len(elements) > 1
    assert all("source_element" in element.meta for element in elements)
    chunks = chunk_elements(
        elements,
        ChunkSpec(target_tokens=30, overlap_tokens=0, max_tokens=40),
        count=lambda text: len(text.split()),
        embed_max=40,
    )
    assert len(chunks) < len(elements)
    for chunk in chunks:
        assert "source_element" not in chunk.meta
        for span in chunk.meta["source_spans"]:
            original = elements[span["element"]]
            assert original.text[span["element_start"] : span["element_end"]] == span["text"]
            assert span["text"] in chunk.text


def test_different_extraction_routes_do_not_share_a_chunk():
    native = Element("Native evidence.", "text", "pdf_text", "text_layer", 0, {"page": 1})
    scanned = Element("Transcribed evidence.", "text", "vision", "scanned_page", 1, {"page": 1})
    chunks = chunk_elements([native, scanned])
    assert [chunk.extractor for chunk in chunks] == ["pdf_text", "vision"]
