"""Excel loader: each sheet -> deterministic markdown table (no LLM).

Embedded images/charts (xlsx only) are best-effort routed to the vision LLM.
"""
from __future__ import annotations

import io
import os

from app.shared.domain.models import Modality
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT
from app.ingest.pipeline.safety import check_image_pixels, check_row_count
from app.ingest.pipeline.tables import df_to_markdown


def extract(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    """Each sheet becomes one TABLE Element; embedded images are best-effort captioned."""
    import pandas as pd

    ext = os.path.splitext(filename or "")[1].lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine=engine)

    els: list[Element] = []
    order = 0
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        check_row_count(len(df), cfg.max_table_rows, kind=f"sheet '{name}'")
        md = df_to_markdown(df)
        if not md:
            continue
        md = f"## {name}\n{md}"       # sheet name as caption (repeated on table chunks)
        els.append(Element(md, Modality.TABLE.value, "xlsx_table", f"sheet:{name}",
                           order, {"sheet": name, "rows": int(len(df))}))
        order += 1

    if ext == ".xlsx":
        order = _extract_embedded_images(data, gateway, els, order, cfg)
    return els


def _extract_embedded_images(data: bytes, gateway, els: list[Element], order: int, cfg) -> int:
    """Best-effort: charts/pictures embedded in the workbook -> vision caption."""
    from app.shared.gateway.client import GatewayError
    from app.ingest.pipeline.safety import UnsafeContentError

    try:
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(data))
        for ws in wb.worksheets:
            for img in getattr(ws, "_images", []) or []:
                blob = img._data() if hasattr(img, "_data") else None
                if not blob:
                    continue
                check_image_pixels(blob, cfg.max_image_pixels)
                caption = gateway.vision(blob, STRICT_TRANSCRIBE_PROMPT, "image/png").strip()
                if not caption:
                    continue
                new = split_blocks(
                    caption, text_extractor="vision", table_extractor="vision_table",
                    text_reason="embedded_image", table_reason="embedded_image_table",
                    text_modality=Modality.IMAGE.value, meta={"sheet": ws.title})
                els.extend(new)
                order += len(new)
    except (GatewayError, UnsafeContentError):
        raise  # gateway failure or an oversized image should fail the job
    except Exception:
        pass  # unreadable embedded objects are skipped, not fatal
    return order
