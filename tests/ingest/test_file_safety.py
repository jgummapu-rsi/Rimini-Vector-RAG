"""File-safety limits: a malicious or malformed upload must be rejected before
it can exhaust worker memory/CPU -- page-count caps, image-pixel caps, and a
soft per-parse timeout."""

import io
import multiprocessing

import pytest
from fpdf import FPDF
from PIL import Image

from app.ingest.pipeline.loaders import extract_document
from app.ingest.pipeline.parse_process import extract_bounded
from app.ingest.pipeline.safety import (
    UnsafeContentError,
    check_image_pixels,
)
from app.shared.config import Settings
from tests.conftest import FakeGateway, NoGateway


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return buf.getvalue()


def _pdf_with_pages(n: int) -> bytes:

    pdf = FPDF()
    for pg in range(n):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 8, f"Page {pg + 1}")
    return bytes(pdf.output())


def test_check_image_pixels_allows_a_small_image():
    check_image_pixels(_png(10, 10), max_pixels=1000)


def test_check_image_pixels_rejects_an_oversized_image():
    with pytest.raises(UnsafeContentError, match="exceeds"):
        check_image_pixels(_png(200, 200), max_pixels=1000)


def test_check_image_pixels_rejects_unreadable_bytes():
    with pytest.raises(UnsafeContentError, match="unreadable"):
        check_image_pixels(b"not an image", max_pixels=1_000_000)


def test_image_loader_rejects_an_oversized_image():
    cfg = Settings(max_image_pixels=100)
    with pytest.raises(UnsafeContentError):
        extract_document("pic.png", _png(50, 50), NoGateway(), cfg)


def test_image_loader_accepts_an_image_within_the_cap():
    cfg = Settings(max_image_pixels=10_000)
    gw = FakeGateway("a small photo")
    els, _ = extract_document("pic.png", _png(10, 10), gw, cfg)
    assert gw.vision_calls == 1
    assert els


def test_pdf_loader_rejects_too_many_pages():
    cfg = Settings(max_pdf_pages=2)
    with pytest.raises(UnsafeContentError, match="page limit"):
        extract_document("big.pdf", _pdf_with_pages(5), NoGateway(), cfg)


def test_pdf_loader_accepts_a_document_within_the_page_cap():
    cfg = Settings(max_pdf_pages=10)
    els, _ = extract_document("small.pdf", _pdf_with_pages(2), NoGateway(), cfg)
    assert els


def test_bounded_parse_returns_the_result_when_fast_enough():
    cfg = Settings(_env_file=None)
    elements, _ = extract_bounded(
        "a.txt", b"Account 00123", NoGateway(), cfg, lambda key: None, lambda key, value: None
    )
    assert elements[0].text == "Account 00123"


def test_bounded_parse_timeout_leaves_no_native_child():

    cfg = Settings(_env_file=None, parse_timeout_seconds=0.001)
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(TimeoutError):
        extract_bounded(
            "a.txt", b"Account 00123", NoGateway(), cfg, lambda key: None, lambda key, value: None
        )
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_bounded_parse_reports_malformed_file():
    with pytest.raises(UnsafeContentError, match="Native parsing failed"):
        extract_bounded(
            "bad.pdf",
            b"not a PDF",
            NoGateway(),
            Settings(_env_file=None),
            lambda key: None,
            lambda key, value: None,
        )
