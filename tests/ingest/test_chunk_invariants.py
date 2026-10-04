from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.chunker import (
    ChunkSpec,
    _compact_table_records,
    _hard_split,
    chunk_elements,
)
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.tokens import count_tokens


def test_wide_table_keeps_all_cells_with_identity_and_budget():
    headers = [f"Field{i}" for i in range(55)]
    values = [f"00{i:03d}" for i in range(55)]
    table = (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * 55)
        + " |\n| "
        + " | ".join(values)
        + " |"
    )
    chunks = chunk_elements([Element(table, "table", "csv", "table", 0)], ChunkSpec())
    assert all(count_tokens(chunk.text) <= 220 for chunk in chunks)
    for name, value in zip(headers, values, strict=False):
        assert any(
            name in chunk.text and value in chunk.text and "Row 1" in chunk.text for chunk in chunks
        )


def test_continuations_keep_nearest_heading():
    source = Element(
        "Evidence about invoices and purchase orders. " * 100,
        "text",
        "text",
        "text",
        0,
        {"section_path": "# Operations > ## Invoice approval"},
    )
    chunks = chunk_elements([source], ChunkSpec(target_tokens=40, max_tokens=60, min_tokens=4))
    assert len(chunks) > 1
    assert all(
        "Invoice approval" in chunk.text and count_tokens(chunk.text) <= 60 for chunk in chunks
    )


def test_page_provenance_never_claims_other_pages():
    elements = [
        Element(
            (f"Page {page} evidence. " * 40),
            "text",
            "pdf_text",
            "text_layer",
            page,
            {"page": page, "bbox": [0, 0, 100, 100]},
        )
        for page in (1, 2)
    ]
    chunks = chunk_elements(elements, ChunkSpec(target_tokens=40, max_tokens=60, min_tokens=4))
    for chunk in chunks:
        assert chunk.meta["pages"] == [chunk.meta["page"]]
        assert f"Page {chunk.meta['page']}" in chunk.text


def test_cjk_and_oversized_heading_are_bounded():
    element = Element(
        "金额账户编号" * 300,
        "text",
        "text",
        "text",
        0,
        {"section_path": "# " + "非常长的标题" * 100},
    )
    chunks = chunk_elements([element], ChunkSpec(target_tokens=40, max_tokens=60, min_tokens=4))
    assert chunks and all(count_tokens(chunk.text) <= 60 for chunk in chunks)


def test_code_is_separate_from_adjacent_prose():
    elements = split_blocks(
        "# API\n\nBefore.\n\n```python\nprint('00123')\n```\n\nAfter.",
        text_extractor="md",
        table_extractor="table",
        text_reason="text",
        table_reason="table",
    )
    chunks = chunk_elements(elements)
    code = [chunk for chunk in chunks if chunk.route_reason == "code_block"]
    assert len(code) == 1 and "00123" in code[0].text and "# API" in code[0].text
    assert "After." not in code[0].text and "Before." not in code[0].text


def test_long_cell_value_retains_its_header_and_row_key_on_every_fragment():
    value = "编号金额" * 200
    table = f"| ID | Details |\n| --- | --- |\n| 00123 | {value} |"
    chunks = chunk_elements([Element(table, "table", "csv", "table", 0)])
    details = [chunk for chunk in chunks if "(Details)" in chunk.text]
    assert len(details) > 1
    assert all("key 00123" in chunk.text and "Row 1" in chunk.text for chunk in details)
    assert "".join(chunk.text.split(": ", 1)[1] for chunk in details) == value


def test_oversized_header_is_preserved_with_linked_cell_records():
    header = "编号" * 300
    table = f"| ID | {header} |\n| --- | --- |\n| 00123 | 007 |"
    chunks = chunk_elements([Element(table, "table", "csv", "table", 0, {"page": 3})])
    assert all(chunk.token_count <= 220 and chunk.meta["page"] == 3 for chunk in chunks)
    assert len({chunk.text.split(";", 1)[0] for chunk in chunks}) == 1
    assert (
        "".join(
            chunk.text.split(": ", 1)[1] for chunk in chunks if "; column 2 header: " in chunk.text
        )
        == header
    )
    assert any("Row 1; column 1: 00123" in chunk.text for chunk in chunks)
    assert any("Row 1; column 2: 007" in chunk.text for chunk in chunks)


def test_hard_split_does_not_add_space_past_token_budget():
    text = "abcd efgh ijkl"
    parts = _hard_split(text, 4, len)
    assert "".join(parts) == text
    assert all(len(part) <= 4 for part in parts)


def test_oversized_caption_and_row_key_are_lossless_with_heading_budget():
    caption, key = "Long caption " * 100, "Long key value " * 100
    parts = _compact_table_records(
        "| ID | Amount |", [[key, "007.00"]], ChunkSpec(max_tokens=60), count_tokens, caption
    )
    assert all(count_tokens(part) <= 60 for part in parts)
    for label, original in [("caption", caption), ("Row 1; column 1", key)]:
        assert (
            "".join(part.split(": ", 1)[1] for part in parts if f"; {label}: " in part) == original
        )
    table = f"{caption}\n| ID | Amount |\n| --- | --- |\n| {key} | 007.00 |"
    chunks = chunk_elements(
        [
            Element(
                table,
                "table",
                "pdf_table",
                "table",
                0,
                {"section_path": "Financial report", "page": 2},
            )
        ]
    )
    assert all(chunk.token_count <= 220 and chunk.meta["page"] == 2 for chunk in chunks)
    assert any("007.00" in chunk.text for chunk in chunks)


def test_exact_source_spans_select_the_original_evidence():
    body = " ".join(f"Unique sentence {index}." for index in range(40))
    source = Element(
        body,
        "text",
        "pdf_text",
        "text_layer",
        0,
        {"page": 2, "source_spans": [{"element": 0, "start": 0, "end": len(body), "page": 2}]},
    )
    chunks = chunk_elements([source], ChunkSpec(target_tokens=40, max_tokens=60, min_tokens=4))
    for chunk in chunks:
        for span in chunk.meta["source_spans"]:
            assert 0 <= span["start"] < span["end"] <= len(body)
            if "precision" not in span:
                assert body[span["start"] : span["end"]] == chunk.text


def test_chunks_keep_only_intersecting_pdf_line_regions():
    lines = [f"Unique evidence line {index} has enough detail." for index in range(20)]
    body = "\n".join(lines)
    spans = []
    offset = 0
    for index, line in enumerate(lines):
        spans.append(
            {
                "element": 0,
                "start": offset,
                "end": offset + len(line),
                "page": 1,
                "bbox": [0, index * 10, 100, index * 10 + 8],
                "text": line,
                "precision": "exact_text",
            }
        )
        offset += len(line) + 1
    source = Element(body, "text", "vision", "scanned_page", 0, {"page": 1, "source_spans": spans})

    chunks = chunk_elements(
        [source], ChunkSpec(target_tokens=30, overlap_tokens=0, max_tokens=40, min_tokens=4)
    )

    assert len(chunks) > 1
    for chunk in chunks:
        chunk_lines = {line for line in lines if line in chunk.text}
        cited_lines = {span["text"] for span in chunk.meta["source_spans"]}
        assert cited_lines == chunk_lines
        assert all(
            body[span["start"] : span["end"]] == span["text"] for span in chunk.meta["source_spans"]
        )


def test_large_code_preserves_whitespace_and_exact_source_sequence():
    code = "```python\n" + "\n".join(f"    account_{i} = '00{i:03d}'" for i in range(80)) + "\n```"
    element = Element(
        code,
        "text",
        "markdown",
        "code_block",
        0,
        {"block_type": "code", "source_spans": [{"element": 0, "start": 0, "end": len(code)}]},
    )
    chunks = chunk_elements([element], ChunkSpec(target_tokens=40, max_tokens=60, min_tokens=4))
    assert "".join(chunk.text for chunk in chunks) == code
    assert all(
        count_tokens(chunk.text) <= 60 and chunk.meta["block_type"] == "code" for chunk in chunks
    )


def test_large_embedding_window_does_not_expand_chunks_to_thousands_of_tokens():
    spec = ChunkSpec.auto(8191)
    assert spec.max_tokens <= 476
    assert spec.target_tokens <= 390
