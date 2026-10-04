"""Excel loader: each sheet -> deterministic markdown table (no LLM).

Embedded images/charts (xlsx only) are best-effort routed to the vision LLM.
"""

from __future__ import annotations

import io
import logging
import os
import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

import xlrd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from PIL import Image

from app.ingest.pipeline.blocks import split_blocks
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.outcomes import omission
from app.ingest.pipeline.prompts import STRICT_TRANSCRIBE_PROMPT
from app.ingest.pipeline.safety import UnsafeContentError, check_image_pixels
from app.ingest.pipeline.tables import rows_to_markdown
from app.shared.domain.models import Modality
from app.shared.gateway.client import GatewayError

log = logging.getLogger(__name__)


def extract(data: bytes, filename: str, gateway, cfg) -> list[Element]:
    """Each sheet becomes one TABLE Element; embedded images are best-effort captioned."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".xlsx":
        return _extract_workbook(data, gateway, cfg)
    return _extract_legacy_workbook(data, cfg)


def _extract_legacy_workbook(data: bytes, cfg) -> list[Element]:

    workbook = xlrd.open_workbook(file_contents=data, formatting_info=True, on_demand=True)
    elements = []
    total_cells = 0
    try:
        if workbook.nsheets > cfg.max_workbook_sheets:
            raise UnsafeContentError("Workbook exceeds the sheet limit")
        for sheet in workbook.sheets():
            total_cells += sheet.nrows * sheet.ncols
            if sheet.nrows > cfg.max_table_rows or total_cells > cfg.max_workbook_cells:
                raise UnsafeContentError("Workbook exceeds the row/cell limit")
            rows = []
            cells = []
            for row in range(sheet.nrows):
                values = []
                for column in range(sheet.ncols):
                    cell = sheet.cell(row, column)
                    value = cell.value
                    if cell.ctype == xlrd.XL_CELL_DATE:
                        value = xlrd.xldate_as_datetime(value, workbook.datemode)
                    number_format = workbook.format_map[
                        workbook.xf_list[cell.xf_index].format_key
                    ].format_str
                    rendered = _display_value(value, number_format)
                    values.append(rendered)
                    cells.append(
                        {
                            "row": row + 1,
                            "column": column + 1,
                            "typed_value": value.isoformat()
                            if isinstance(value, (date, datetime))
                            else value,
                            "number_format": number_format,
                            "rendered_value": rendered,
                            "formula_status": "cached_value_only",
                        }
                    )
                rows.append(values)
            if rows:
                elements.append(
                    Element(
                        f"## {sheet.name}\n{rows_to_markdown(rows)}",
                        "table",
                        "xls_table",
                        "legacy_sheet",
                        len(elements),
                        {
                            "sheet": sheet.name,
                            "cells": cells,
                            "cell_range": _cell_range(sheet.nrows, sheet.ncols),
                        },
                    )
                )
                elements.append(
                    omission(
                        "xls",
                        "legacy_formula_and_display_fidelity_unverified",
                        len(elements),
                        sheet=sheet.name,
                    )
                )
    finally:
        workbook.release_resources()
    return elements


def _display_value(value, number_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        formats = {
            "yyyy-mm-dd": "%Y-%m-%d",
            "mm/dd/yyyy": "%m/%d/%Y",
            "dd/mm/yyyy": "%d/%m/%Y",
            "yyyy-mm-dd hh:mm:ss": "%Y-%m-%d %H:%M:%S",
        }
        if number_format.lower() in formats:
            return value.strftime(formats[number_format.lower()])
        return value.isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if re.fullmatch(r"0+", number_format):
            return f"{int(value):0{len(number_format)}d}" if value == int(value) else str(value)
        numeric = re.fullmatch(r"([\$€£]?)(#,##0|0)(?:\.(0+))?(%)?", number_format)
        if numeric:
            currency, integer, decimals, percent = numeric.groups()
            places = len(decimals or "")
            amount = Decimal(str(value)) * (100 if percent else 1)
            rounded = amount.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
            return (
                currency
                + format(rounded, f"{',' if ',' in integer else ''}.{places}f")
                + (percent or "")
            )
    return str(value)


def _extract_workbook(data: bytes, gateway, cfg) -> list[Element]:

    source = load_workbook(io.BytesIO(data), data_only=False)
    cached = load_workbook(io.BytesIO(data), data_only=True)
    elements = []
    try:
        if len(source.worksheets) > cfg.max_workbook_sheets:
            raise UnsafeContentError("Workbook exceeds the sheet limit")
        total_cells = 0
        for sheet in source.worksheets:
            total_cells += sheet.max_row * sheet.max_column
            if sheet.max_row > cfg.max_table_rows or total_cells > cfg.max_workbook_cells:
                raise UnsafeContentError("Workbook exceeds the row/cell limit")
            rows = []
            cells = []
            for row in sheet.iter_rows():
                display_row = []
                for cell in row:
                    value = cell.value
                    formula = str(value) if cell.data_type == "f" else None
                    typed = cached[sheet.title][cell.coordinate].value if formula else value
                    display = _display_value(typed, cell.number_format)
                    format_supported = (
                        not isinstance(typed, (int, float, date, datetime))
                        or isinstance(typed, bool)
                        or cell.number_format == "General"
                        or bool(re.fullmatch(r"[\$€£]?(?:#,##0|0+)(?:\.0+)?%?", cell.number_format))
                        or (
                            isinstance(typed, (date, datetime))
                            and cell.number_format.lower()
                            in {"yyyy-mm-dd", "mm/dd/yyyy", "dd/mm/yyyy", "yyyy-mm-dd hh:mm:ss"}
                        )
                    )
                    if formula and typed is None:
                        display = f"Formula {formula} (cached value unavailable)"
                    display_row.append(display)
                    cells.append(
                        {
                            "address": cell.coordinate,
                            "typed_value": typed.isoformat()
                            if isinstance(typed, (date, datetime))
                            else typed,
                            "row": cell.row,
                            "column": cell.column,
                            "display_value": display if format_supported else None,
                            "rendered_value": display,
                            "display_status": "exact" if format_supported else "typed_value_only",
                            "formula": formula,
                            "number_format": cell.number_format,
                        }
                    )
                rows.append(display_row)
            if any(any(value for value in row) for row in rows):
                elements.append(
                    Element(
                        f"## {sheet.title}\n{rows_to_markdown(rows)}",
                        "table",
                        "xlsx_table",
                        f"sheet:{sheet.title}",
                        len(elements),
                        {
                            "sheet": sheet.title,
                            "rows": max(0, len(rows) - 1),
                            "cells": cells,
                            "cell_range": _cell_range(sheet.max_row, sheet.max_column),
                        },
                    )
                )
            unsupported = [
                cell["address"] for cell in cells if cell["display_status"] == "typed_value_only"
            ]
            if unsupported:
                elements.append(
                    omission(
                        "xlsx_format",
                        "unsupported_display_format",
                        len(elements),
                        sheet=sheet.title,
                        cells=unsupported,
                    )
                )
            for chart in sheet._charts:
                elements.append(
                    Element(
                        "",
                        "image",
                        "xlsx_chart",
                        "unsupported_chart",
                        len(elements),
                        {
                            "sheet": sheet.title,
                            "omission": "native_chart_not_rendered",
                            "chart_type": type(chart).__name__,
                        },
                    )
                )
            _extract_embedded_images(sheet, gateway, elements, cfg)
    finally:
        source.close()
        cached.close()
    return elements


def _extract_embedded_images(sheet, gateway, elements: list[Element], cfg) -> None:

    omitted = 0
    for index, image in enumerate(sheet._images):
        anchor = getattr(image.anchor, "_from", None)
        meta = {"sheet": sheet.title, "image_index": index}
        if anchor is not None:
            meta.update(row=anchor.row + 1, column=anchor.col + 1)
        try:
            blob = image._data()
            check_image_pixels(blob, cfg.max_image_pixels)
            buffer = io.BytesIO()
            with Image.open(io.BytesIO(blob)) as decoded:
                decoded.convert("RGB").save(buffer, format="PNG")
            caption = gateway.vision(
                buffer.getvalue(), STRICT_TRANSCRIBE_PROMPT, "image/png"
            ).strip()
            if not caption:
                raise UnsafeContentError("Workbook image returned no evidence")
            new = split_blocks(
                caption,
                text_extractor="vision",
                table_extractor="vision_table",
                text_reason="embedded_image",
                table_reason="embedded_image_table",
                text_modality=Modality.IMAGE.value,
                meta=meta,
            )
            for element in new:
                element.order = len(elements)
                elements.append(element)
        except GatewayError:
            raise
        except (ValueError, OSError, KeyError, AttributeError) as exc:
            elements.append(omission("xlsx_image", type(exc).__name__, len(elements), **meta))
            omitted += 1
    if omitted:
        log.warning(
            "Workbook images could not be extracted",
            extra={"event": "workbook_image_omissions", "omitted_regions": omitted},
        )


def _cell_range(rows: int, columns: int) -> str:
    if rows <= 0 or columns <= 0:
        return ""
    return f"A1:{get_column_letter(columns)}{rows}"
