"""File-safety limits: a malicious or malformed upload must be rejected before
it can exhaust worker memory/CPU -- page-count caps, image-pixel caps, table
row caps, and a soft per-parse timeout."""
import io
import time

import pytest
from PIL import Image

from app.shared.config import Settings
from app.ingest.pipeline.loaders import extract_document
from app.ingest.pipeline.safety import (
    UnsafeContentError,
    check_image_pixels,
    check_row_count,
    run_with_timeout,
)
from tests.conftest import NoGateway


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return buf.getvalue()


def _pdf_with_pages(n: int) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    for pg in range(n):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 8, f"Page {pg + 1}")
    return bytes(pdf.output())


# ------------------------------------------------------------ image pixels --


def test_check_image_pixels_allows_a_small_image():
    check_image_pixels(_png(10, 10), max_pixels=1000)  # 100px, well under cap


def test_check_image_pixels_rejects_an_oversized_image():
    with pytest.raises(UnsafeContentError, match="exceeds"):
        check_image_pixels(_png(200, 200), max_pixels=1000)  # 40,000px > 1,000


def test_check_image_pixels_rejects_unreadable_bytes():
    with pytest.raises(UnsafeContentError, match="unreadable"):
        check_image_pixels(b"not an image", max_pixels=1_000_000)


def test_image_loader_rejects_an_oversized_image():
    cfg = Settings(max_image_pixels=100)  # 10x10 = 100px is the ceiling
    with pytest.raises(UnsafeContentError):
        extract_document("pic.png", _png(50, 50), NoGateway(), cfg)  # 2,500px


def test_image_loader_accepts_an_image_within_the_cap():
    from tests.conftest import FakeGateway
    cfg = Settings(max_image_pixels=10_000)
    gw = FakeGateway("a small photo")
    els, _ = extract_document("pic.png", _png(10, 10), gw, cfg)
    assert gw.vision_calls == 1
    assert els


# -------------------------------------------------------------- row counts --


def test_check_row_count_allows_within_the_cap():
    check_row_count(50, max_rows=100)


def test_check_row_count_rejects_over_the_cap():
    with pytest.raises(UnsafeContentError, match="exceeds"):
        check_row_count(101, max_rows=100)


def test_csv_loader_rejects_too_many_rows():
    header = "id,value\n"
    rows = "\n".join(f"{i},{i}" for i in range(50))
    csv_bytes = (header + rows).encode()
    cfg = Settings(max_table_rows=10)  # 50 data rows > 10
    with pytest.raises(UnsafeContentError):
        extract_document("big.csv", csv_bytes, NoGateway(), cfg)


def test_csv_loader_accepts_a_table_within_the_row_cap():
    header = "id,value\n"
    rows = "\n".join(f"{i},{i}" for i in range(5))
    csv_bytes = (header + rows).encode()
    cfg = Settings(max_table_rows=100)
    els, summ = extract_document("small.csv", csv_bytes, NoGateway(), cfg)
    assert summ["by_modality"] == {"table": 1}


# ------------------------------------------------------------- pdf pages --


def test_pdf_loader_rejects_too_many_pages():
    cfg = Settings(max_pdf_pages=2)
    with pytest.raises(UnsafeContentError, match="page limit"):
        extract_document("big.pdf", _pdf_with_pages(5), NoGateway(), cfg)


def test_pdf_loader_accepts_a_document_within_the_page_cap():
    cfg = Settings(max_pdf_pages=10)
    els, _ = extract_document("small.pdf", _pdf_with_pages(2), NoGateway(), cfg)
    assert els


# ------------------------------------------------------ soft parse timeout --


def test_run_with_timeout_returns_the_result_when_fast_enough():
    assert run_with_timeout(lambda: 42, timeout_seconds=5) == 42


def test_run_with_timeout_raises_when_the_call_hangs():
    with pytest.raises(TimeoutError):
        run_with_timeout(lambda: time.sleep(5), timeout_seconds=0.05)


def test_run_with_timeout_reraises_the_callables_own_exception():
    def boom():
        raise ValueError("bad file")

    with pytest.raises(ValueError, match="bad file"):
        run_with_timeout(boom, timeout_seconds=5)
