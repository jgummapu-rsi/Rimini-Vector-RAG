from app.retrieval.rag.evidence_quality import useful_indices


def meta(page, label="plain text", y=0.1, document="manual"):
    return {
        "document_id": document,
        "generation_id": "v1",
        "provenance": {
            "pages": [page],
            "regions": [{"layout_label": label, "y": y, "height": 0.02}],
        },
    }


def test_repeated_page_furniture_removed_but_body_and_mixed_chunks_retained():
    texts = [
        "Annual manual",
        "Annual manual",
        "Page 1 of 2",
        "Warehouse must stay locked",
        "Warehouse must stay locked",
        "Instructions and footer",
    ]
    metadata = [
        meta(1),
        meta(2),
        meta(1, "abandon", 0.95),
        meta(1, y=0.4),
        meta(2, y=0.4),
        meta(2, y=0.4),
    ]
    metadata[-1]["provenance"]["regions"].append(
        {"layout_label": "abandon", "y": 0.95, "height": 0.02}
    )
    assert useful_indices("How is the warehouse secured?", texts, metadata) == [3, 4, 5]
    assert useful_indices("What is the header?", texts, metadata) == list(range(6))


def test_distinct_documents_and_unlocated_text_are_not_deduplicated():
    texts = ["University X", "University X", "University X"]
    assert useful_indices(
        "Who studied here?", texts, [meta(1, document="a"), meta(2, document="b"), {}]
    ) == [0, 1, 2]


def test_logo_filter_is_query_aware_and_preserves_body_figures():
    texts = ["Company logo: yellow wordmark", "Company logo: yellow wordmark"]
    metadata = [meta(1, "figure", 0.02), meta(2, "figure", 0.4)]
    assert useful_indices("How are shipments handled?", texts, metadata) == [1]
    assert useful_indices("Describe the logo", texts, metadata) == [0, 1]
