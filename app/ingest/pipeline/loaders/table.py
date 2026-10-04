"""Standalone table loaders: CSV and HTML tables (deterministic markdown, no LLM)."""

from __future__ import annotations

import io
import logging

import pandas as pd
from lxml import html

from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.outcomes import omission
from app.ingest.pipeline.safety import UnsafeContentError
from app.ingest.pipeline.tables import df_to_markdown, rows_to_markdown
from app.shared.domain.models import Modality

log = logging.getLogger(__name__)


def extract_csv(data: bytes, filename: str, gateway, cfg) -> list[Element]:

    try:
        df = pd.read_csv(
            io.BytesIO(data),
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=cfg.max_table_rows + 1,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise UnsafeContentError("CSV could not be parsed without losing source values") from exc
    if len(df) > cfg.max_table_rows:
        raise UnsafeContentError("CSV exceeds the row limit")

    md = df_to_markdown(df)
    if not md:
        return []
    return [
        Element(
            md,
            Modality.TABLE.value,
            "csv_table",
            "structured_csv",
            0,
            {"rows": int(len(df)), "row_start": 1, "row_end": int(len(df))},
        )
    ]


def extract_html(data: bytes, filename: str, gateway, cfg) -> list[Element]:

    root = html.fromstring(data.decode("utf-8-sig"))
    elements = []
    headings = []
    table_index = 0

    def emit(value, modality="text", **meta):
        value = value.strip()
        if value:
            elements.append(
                Element(
                    value,
                    modality,
                    "html_table" if modality == "table" else "html",
                    "structured_html" if modality == "table" else "html_prose",
                    len(elements),
                    {"section_path": " > ".join(t for _, t in headings), **meta},
                )
            )

    def walk(node):
        nonlocal table_index
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag in {"script", "style", "head"}:
            return
        if tag in {"img", "svg", "canvas", "iframe", "object"}:
            elements.append(
                omission(
                    "html",
                    "unsupported_visual_object",
                    len(elements),
                    source_line=node.sourceline,
                    object_type=tag,
                )
            )
            if node.get("alt"):
                emit(node.get("alt"), source_line=node.sourceline)
            return
        if tag == "table":
            rows = _html_rows(node)
            emit(
                rows_to_markdown(rows),
                "table",
                table_index=table_index,
                rows=max(0, len(rows) - 1),
                row_start=1,
                row_end=max(0, len(rows) - 1),
                source_line=node.sourceline,
            )
            table_index += 1
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            value = "#" * level + " " + "".join(node.itertext()).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, value))
            emit(value, source_line=node.sourceline)
            return
        if tag in {"p", "li", "pre", "blockquote"} and not node.xpath(".//table"):
            emit("".join(node.itertext()), source_line=node.sourceline)
            return
        emit(node.text or "", source_line=node.sourceline)
        for child in node:
            walk(child)
            emit(child.tail or "", source_line=child.sourceline)

    walk(root)
    return elements


def _html_rows(table) -> list[list[str]]:
    rows = []
    spans = {}
    for row_index, row in enumerate(table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")):
        values = {}
        for (target_row, column), value in spans.items():
            if target_row == row_index:
                values[column] = value
        column = 0
        for cell in row.xpath("./th|./td"):
            while column in values:
                column += 1
            value = "".join(cell.itertext())
            width = int(cell.get("colspan", "1"))
            height = int(cell.get("rowspan", "1"))
            if not 1 <= width <= 1000 or not 1 <= height <= 1000:
                raise UnsafeContentError("HTML table span exceeds the supported cell limit")
            for offset in range(width):
                values[column + offset] = value
                for below in range(1, height):
                    spans[row_index + below, column + offset] = value
            column += width
        rows.append([values.get(index, "") for index in range(max(values, default=-1) + 1)])
    return rows
