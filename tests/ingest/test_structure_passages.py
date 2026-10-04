from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.loaders.pdf import _annotate_sections, _reading_order


def element(text, box, *, page=1, heading=False):
    return Element(
        text,
        "text",
        "pdf_text",
        "text_layer",
        0,
        {
            "page": page,
            "page_width": 600,
            "page_height": 800,
            "bbox": box,
            "font_size": 12 if heading else 10,
            "font_bold": heading,
        },
    )


def test_label_rows_do_not_become_columns_around_full_width_prose():
    elements = [
        element("Architect", (20, 10, 180, 22)),
        element("Company Alpha", (20, 26, 190, 38)),
        element("2020 - 2022", (450, 10, 580, 22)),
        element("London", (500, 26, 580, 38)),
        element(
            "Built a platform with reliable processing of customer requests across many regions and teams.",
            (20, 45, 580, 57),
        ),
    ]
    ordered = _reading_order([((e.meta["bbox"][1], e.meta["bbox"][0]), e) for e in elements], 600)
    assert [e.text for _, e in ordered][:4] == [
        "Architect",
        "2020 - 2022",
        "Company Alpha",
        "London",
    ]


def test_visible_sections_keep_each_subject_with_its_own_duties_and_continue_across_pages():
    elements = [
        element("Architect", (20, 10, 180, 22), heading=True),
        element("2020 - 2022", (450, 10, 580, 22)),
        element("Company Alpha", (20, 26, 190, 38), heading=True),
        element(
            "Built insurance software with a team of fifteen engineers supporting customers across multiple regions.",
            (20, 45, 580, 57),
        ),
        element("Principal Engineer", (20, 80, 190, 92), heading=True),
        element("Company Beta", (20, 96, 190, 108), heading=True),
        element(
            "Built distributed Spark automation and failover for large clusters saving manual effort every quarter.",
            (20, 120, 580, 132),
        ),
        element("Automated infrastructure setup.", (20, 20, 580, 32), page=2),
    ]
    _annotate_sections(elements)
    chunks = chunk_elements(elements, count=lambda s: len(s.split()))
    assert len(chunks) == 3
    assert "Company Alpha" in chunks[0].meta["section_path"]
    assert "Company Beta" not in chunks[0].text
    assert all("Company Beta" in c.meta["section_path"] for c in chunks[1:])
    assert all("insurance" not in c.text for c in chunks[1:])
    assert chunks[-1].meta["pages"] == [2]


def test_repeated_text_keeps_its_actual_occurrence_geometry():
    text = "Repeated evidence sentence.\n\n" * 4
    phrase = "Repeated evidence sentence."
    spans = [
        {
            "start": i * (len(phrase) + 2),
            "end": i * (len(phrase) + 2) + len(phrase),
            "text": phrase,
            "element": i,
            "bbox": [0, i * 20, 200, i * 20 + 10],
        }
        for i in range(4)
    ]
    source = Element(text, "text", "text", "prose", 0, {"source_spans": spans})
    chunks = chunk_elements(
        [source],
        ChunkSpec(target_tokens=3, max_tokens=6, overlap_tokens=0, min_tokens=1),
        count=lambda s: len(s.split()),
    )
    assert len(chunks) == 4
    assert [c.meta["source_spans"][0]["element"] for c in chunks] == list(range(4))


def test_bullet_items_and_whitespace_survive_packing():
    body = "● First item spans\na wrapped line without punctuation\n● Second item spans\nanother wrapped line"
    chunks = chunk_elements(
        [Element(body, "text", "text", "prose", 0)],
        ChunkSpec(target_tokens=10, max_tokens=14, overlap_tokens=0, min_tokens=1),
        count=lambda s: len(s.split()),
    )
    assert len(chunks) == 2
    assert all(c.text.startswith("●") and c.text in body for c in chunks)
