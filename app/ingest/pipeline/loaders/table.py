"""Standalone table loaders: CSV and HTML tables (deterministic markdown, no LLM)."""
from __future__ import annotations

import io

from app.shared.domain.models import Modality
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.safety import check_row_count
from app.ingest.pipeline.tables import df_to_markdown


def extract_csv(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    """Parse a standalone CSV into one TABLE Element (or a text fallback if unparseable)."""
    import pandas as pd

    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        # not parseable as a table -> fall back to raw text
        content = data.decode("utf-8", "ignore").strip()
        if not content:
            return []
        return [Element(content, Modality.TEXT.value, "text", "csv_fallback_text", 0, {})]

    check_row_count(len(df), cfg.max_table_rows, kind="csv")
    md = df_to_markdown(df)
    if not md:
        return []
    return [Element(md, Modality.TABLE.value, "csv_table", "structured_csv", 0,
                    {"rows": int(len(df))})]


def extract_html(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    """Parse every HTML table into its own TABLE Element."""
    import pandas as pd

    try:
        dfs = pd.read_html(io.BytesIO(data))
    except Exception:
        return []
    els: list[Element] = []
    for i, df in enumerate(dfs):
        check_row_count(len(df), cfg.max_table_rows, kind="html table")
        md = df_to_markdown(df)
        if md:
            els.append(Element(md, Modality.TABLE.value, "html_table", "structured_html",
                               i, {"table_index": i, "rows": int(len(df))}))
    return els
