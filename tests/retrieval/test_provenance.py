from app.retrieval.rag.provenance import citation_provenance
from app.retrieval.rag.query import _quote_provenance, _quotes_provenance


def test_pdf_regions_are_normalized_clipped_and_deduplicated():
    meta = {
        "page": 2,
        "pages": [2],
        "page_width": 200,
        "page_height": 100,
        "source_spans": [
            {
                "element": 4,
                "page": 2,
                "bbox": [-10, 10, 100, 40],
                "page_width": 200,
                "page_height": 100,
            },
            {
                "element": 4,
                "page": 2,
                "bbox": [-10, 10, 100, 40],
                "page_width": 200,
                "page_height": 100,
            },
        ],
    }

    result = citation_provenance(meta, "pdf")

    assert result["kind"] == "pdf"
    assert result["pages"] == [2]
    assert result["regions"] == [
        {
            "page": 2,
            "x": 0.0,
            "y": 0.1,
            "width": 0.5,
            "height": 0.3,
            "source_element": 4,
            "precision": "exact_text",
        }
    ]


def test_structural_locators_are_allowlisted_without_leaking_private_meta():
    result = citation_provenance(
        {
            "sheet": "Revenue",
            "cell_range": "A1:F20",
            "cells": ["large private payload"],
            "secret": "must not escape",
        },
        "xlsx",
    )

    assert result["kind"] == "spreadsheet"
    assert result["locator"] == {"sheet": "Revenue", "cell_range": "A1:F20"}
    assert "cells" not in result["locator"]
    assert "secret" not in result["locator"]


def test_crop_origin_and_outside_regions():
    result = citation_provenance(
        {
            "page": 1,
            "page_width": 600,
            "page_height": 800,
            "page_box": [100, 200, 500, 600],
            "source_spans": [{"bbox": [140, 240, 180, 280]}, {"bbox": [0, 0, 50, 50]}],
        },
        "pdf",
    )
    assert len(result["regions"]) == 1
    assert result["regions"][0]["x"] == 0.1
    assert result["regions"][0]["y"] == 0.1


def test_chunk_outline_survives_quote_selection_and_keeps_pages_separate():
    result = citation_provenance(
        {
            "page_width": 100,
            "page_height": 100,
            "source_spans": [
                {"page": 1, "bbox": [10, 10, 40, 20], "text": "First sentence"},
                {"page": 1, "bbox": [10, 30, 70, 40], "text": "Second sentence"},
                {"page": 2, "bbox": [20, 50, 60, 60], "text": "Another page"},
            ],
        },
        "pdf",
    )
    assert result["chunk_regions"] == [
        {"page": 1, "x": 0.1, "y": 0.1, "width": 0.6, "height": 0.3, "precision": "element_region"},
        {"page": 2, "x": 0.2, "y": 0.5, "width": 0.4, "height": 0.1, "precision": "element_region"},
    ]
    for selected in (
        _quote_provenance(result, "Second sentence"),
        _quotes_provenance(result, ["Second sentence"]),
        _quote_provenance(result, None),
        _quote_provenance(result, "Not present"),
    ):
        assert selected["chunk_regions"] == result["chunk_regions"]
    selected = _quote_provenance(result, "Second sentence")
    assert len(selected["regions"]) == 1
    assert selected["regions"][0]["y"] == 0.3


def test_chunk_outline_includes_regions_beyond_quote_region_limit(monkeypatch):
    monkeypatch.setattr("app.retrieval.rag.provenance._MAX_REGIONS", 1)
    result = citation_provenance(
        {
            "page": 1,
            "page_width": 100,
            "page_height": 100,
            "source_spans": [{"bbox": [10, 10, 20, 20]}, {"bbox": [10, 80, 90, 90]}],
        },
        "pdf",
    )
    assert result["regions_truncated"]
    assert result["chunk_regions"][0]["height"] == 0.8
    assert result["chunk_regions"][0]["width"] == 0.8
