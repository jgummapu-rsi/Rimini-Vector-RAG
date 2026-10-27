import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.loaders.image import extract
from app.retrieval.rag.provenance import citation_provenance
from app.retrieval.rag.query import _quote_provenance


def picture(*, orientation=None):
    buffer = io.BytesIO()
    with Image.new("RGB", (200, 100), "white") as image:
        exif = image.getexif()
        if orientation:
            exif[274] = orientation
        image.save(buffer, "PNG", exif=exif)
    return buffer.getvalue()


def config():
    return SimpleNamespace(max_pdf_pages=5, max_image_pixels=100000)


class Gateway:
    def __init__(self, crops, full_text):
        self.crops = crops
        self.responses = iter([*crops, full_text])
        self.sizes = []

    def layout(self, png):
        with Image.open(io.BytesIO(png)) as image:
            self.sizes.append(image.size)
        return [
            {
                "id": index,
                "bbox": [0.1, 0.1 + index * 0.3, 0.9, 0.3 + index * 0.3],
                "label": "plain text",
                "confidence": 0.95,
            }
            for index in range(len(self.crops))
        ]

    def vision(self, png, prompt, mime):
        with Image.open(io.BytesIO(png)) as image:
            self.sizes.append(image.size)
        return next(self.responses)


def selected(elements, quote):
    chunks = chunk_elements(elements)
    assert len(chunks) == 1
    return _quote_provenance(citation_provenance(chunks[0].meta, "image"), quote)


def test_image_quote_selects_only_matching_crop_after_chunking():
    gateway = Gateway(
        ["Premium support costs 100 USD.", "Basic support costs 20 USD."],
        "Premium support costs 100 USD.\n\nBasic support costs 20 USD.",
    )
    elements = extract(picture(), "support.png", gateway, config())
    result = selected(elements, "Basic support costs 20 USD.")
    assert result["kind"] == "image"
    assert result["locator"]["frame"] == 1
    assert result["selection_status"] == "region"
    assert len(result["regions"]) == 1
    assert result["regions"][0]["y"] == pytest.approx(0.4)
    assert result["regions"][0]["height"] == pytest.approx(0.2)
    assert result["regions"][0]["precision"] == "element_region"


def test_uncovered_text_keeps_frame_geometry_and_reading_order():
    text = "First missed.\n\nKnown block.\n\nLast missed."
    elements = extract(picture(), "scan.png", Gateway(["Known block."], text), config())
    result = selected(elements, "Last missed.")
    assert len(result["regions"]) == 1
    assert result["regions"][0]["precision"] == "page_region"
    assert result["regions"][0]["width"] == 1
    # The two fallback spans share a rectangle but neither may be discarded.
    result = selected(elements, "First missed. Known block. Last missed.")
    assert len(result["regions"]) == 3


@pytest.mark.parametrize(
    "crops,text,quote",
    [
        (["Same wording.", "Same wording."], "Same wording.", "Same wording."),
        (["Total 100 USD"], "Total 1.00 USD", "Total 1.00 USD"),
        (["Total 100 USD"], "Total 100 USD and Total 100 USD", "Total 100 USD"),
        (["platform"], "platforms", "platforms"),
    ],
)
def test_ambiguous_or_inexact_crop_match_never_gets_a_crop_highlight(crops, text, quote):
    elements = extract(picture(), "scan.png", Gateway(crops, text), config())
    spans = elements[0].meta["source_spans"]
    assert all(span["precision"] == "page_region" for span in spans)
    result = selected(elements, quote)
    assert all(region["precision"] == "page_region" for region in result["regions"])


def test_no_detected_regions_still_cites_frame_without_invented_word_boxes():
    gateway = Gateway([], "Source evidence")
    result = selected(extract(picture(), "scan.png", gateway, config()), "Source evidence")
    assert gateway.sizes == [(200, 100), (200, 100)]
    assert result["selection_status"] == "region"
    assert result["regions"][0]["precision"] == "page_region"


def test_orientation_is_applied_before_layout_crops_and_transcription():
    gateway = Gateway(["Rotated evidence"], "Rotated evidence")
    elements = extract(picture(orientation=6), "rotated.png", gateway, config())
    assert gateway.sizes == [(100, 200), (80, 40), (100, 200)]
    assert elements[0].meta["page_width"] == 100
    assert elements[0].meta["page_height"] == 200
    assert selected(elements, "Rotated evidence")["regions"][0]["height"] == 0.2


def test_tiff_spans_and_quotes_stay_on_their_frame():
    buffer = io.BytesIO()
    with Image.new("RGB", (200, 100), "white") as first, Image.new("RGB", (100, 200)) as second:
        first.save(buffer, "TIFF", save_all=True, append_images=[second])
    gateway = Gateway([], "unused")
    gateway.responses = iter(["First frame evidence", "Second frame evidence"])
    elements = extract(buffer.getvalue(), "scan.tiff", gateway, config())
    result = selected(elements[1:], "Second frame evidence")
    assert result["pages"] == [2]
    assert result["locator"]["frame"] == 2
    assert result["regions"][0]["page"] == 2


def test_dense_image_chunks_only_carry_their_own_text_ranges():
    text = "\n\n".join(f"Paragraph {index} contains its own evidence." for index in range(30))
    elements = extract(picture(), "dense.png", Gateway([], text), config())
    chunks = chunk_elements(
        elements,
        ChunkSpec(target_tokens=40, max_tokens=50, min_tokens=1, overlap_tokens=0),
    )
    assert len(chunks) > 1
    for chunk in chunks:
        spans = chunk.meta["source_spans"]
        assert all(span["text"] in chunk.text for span in spans)
        result = _quote_provenance(citation_provenance(chunk.meta, "image"), chunk.text)
        assert result["selection_status"] == "region"


def test_missing_quote_never_creates_selected_regions():
    elements = extract(picture(), "scan.png", Gateway(["Evidence"], "Evidence"), config())
    result = selected(elements, None)
    assert result["selection_status"] == "missing_quote"
    assert result["regions"] == []


def test_repeated_text_in_separate_blocks_does_not_reuse_a_unique_crop():
    table = "| Item | Cost |\n| --- | --- |\n| Support | 100 |"
    text = f"{table}\n\nOther text.\n\n{table}"
    elements = extract(picture(), "scan.png", Gateway([table], text), config())
    assert len(elements) == 3
    assert all(
        span["precision"] == "page_region"
        for element in elements
        for span in element.meta["source_spans"]
    )


def test_image_table_retains_region_geometry():
    table = "| Item | Cost |\n| --- | --- |\n| Support | 100 |"
    elements = extract(picture(), "table.png", Gateway([table], table), config())
    assert elements[0].modality == "table"
    result = selected(elements, "Support | 100")
    assert result["selection_status"] == "region"
    assert result["regions"][0]["precision"] == "element_region"
