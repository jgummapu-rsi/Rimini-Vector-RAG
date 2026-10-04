"""Exercise real layout inference through the bounded PDF ingestion worker."""

from __future__ import annotations

from fpdf import FPDF

from app.ingest.pipeline.parse_process import extract_bounded
from app.shared.config import Settings


def main():
    document = FPDF()
    document.add_page()
    document.set_font("Helvetica", size=18)
    document.cell(0, 12, "Support policy", new_x="LMARGIN", new_y="NEXT")
    document.set_font("Helvetica", size=12)
    sentence = "Premium customers receive priority support during business hours. "
    document.multi_cell(0, 8, sentence * 8)
    cfg = Settings(_env_file=None, layout_enabled=True, parse_timeout_seconds=120)
    artifacts = {}
    elements, summary = extract_bounded(
        "layout-check.pdf",
        bytes(document.output()),
        None,
        cfg,
        artifacts.get,
        artifacts.__setitem__,
    )
    words = [span for element in elements for span in element.meta.get("source_spans", [])]
    assert words and all(span["precision"] == "word" for span in words)
    assert any("layout_id" in span for span in words), "Detector did not anchor native words"
    again, reused = extract_bounded(
        "layout-check.pdf",
        bytes(document.output()),
        None,
        cfg,
        artifacts.get,
        artifacts.__setitem__,
    )
    assert elements == again and reused["extraction_reused"]
    print(
        f"Bounded layout ingestion passed: {len(words)} words, "
        f"{len(artifacts)} artifacts, {summary['vision_calls']} vision calls; "
        "retry reused extraction"
    )


if __name__ == "__main__":
    main()
