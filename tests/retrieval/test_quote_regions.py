from app.retrieval.rag.provenance import citation_provenance
from app.retrieval.rag.query import _quote_provenance


def region(text, page=1):
    return {"text": text, "page": page, "x": 0.1, "y": 0.2, "width": 0.7, "height": 0.02}


def test_quote_excludes_later_lines_that_share_common_words():
    lines = [
        region("Build a workflow with tools."),
        region("Build a different workflow with tools."),
    ]
    selected = _quote_provenance({"regions": lines}, "Build a workflow with tools.")
    assert selected["regions"] == lines[:1]
    assert selected["selection_status"] == "quote"


def test_quote_crosses_hyphenated_line_break_and_ignores_markdown():
    lines = [
        region("Generative AI experi-"),
        region("ence with **Python**."),
        region("Unrelated section"),
    ]
    selected = _quote_provenance({"regions": lines}, "AI experience with Python")
    assert selected["regions"] == lines[:2]


def test_ambiguous_quote_does_not_guess_an_occurrence():
    lines = [region("Annual total"), region("Annual total", page=2)]
    selected = _quote_provenance({"regions": lines}, "Annual total")
    assert selected["regions"] == []
    assert selected["selection_status"] == "unmapped_quote"


def test_unmatched_quote_does_not_highlight_all_chunk_regions():
    selected = _quote_provenance({"regions": [region("Unrelated source line")]}, "Different quote")
    assert selected["regions"] == []


def test_no_quote_does_not_paint_the_entire_chunk():
    lines = [region("Source line")]
    assert _quote_provenance({"regions": lines}, None) == {
        "regions": [],
        "selection_status": "missing_quote",
    }


def test_coarse_quote_never_becomes_precise():
    selected = _quote_provenance(
        {"regions": [dict(region("Invoice total"), precision="page_region")]}, "Invoice total"
    )
    assert selected["selection_status"] == "region"


def test_quote_after_twenty_four_regions_is_retained():
    spans = [
        {
            "page": 1,
            "bbox": [0, index * 10, 100, index * 10 + 8],
            "text": f"Unique evidence {index}",
        }
        for index in range(40)
    ]
    provenance = citation_provenance(
        {"page_width": 500, "page_height": 500, "source_spans": spans}, "pdf"
    )
    result = _quote_provenance(provenance, "Unique evidence 39")
    assert result["regions"][0]["text"] == "Unique evidence 39"


def test_quote_does_not_match_inside_another_word_or_erase_decimal_points():
    lines = [region("platform"), region("platforms")]
    assert _quote_provenance({"regions": lines}, "platform")["regions"] == lines[:1]
    assert _quote_provenance({"regions": [region("125 USD")]}, "1.25 USD")["regions"] == []
