from app.retrieval.rag.provenance import citation_provenance


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
