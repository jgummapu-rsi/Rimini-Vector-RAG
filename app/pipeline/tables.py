"""Deterministic table -> GitHub-flavored markdown (no LLM).

Used for every *structured* table source: CSV, Excel sheets, DOCX tables, and
PDF text-layer tables. The LLM is only involved when a table exists as an image.
"""
from __future__ import annotations

import re
from typing import Any, Sequence


def _esc(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    return (
        s.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def rows_to_markdown(rows: Sequence[Sequence[Any]]) -> str:
    """First row is treated as the header. Ragged rows are padded."""
    rows = [r for r in rows if r is not None]
    if not rows:
        return ""
    ncol = max((len(r) for r in rows), default=0)
    if ncol == 0:
        return ""

    def norm(r: Sequence[Any]) -> list[str]:
        return [_esc(r[i]) if i < len(r) else "" for i in range(ncol)]

    header = norm(rows[0])
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncol) + " |",
    ]
    for r in rows[1:]:
        lines.append("| " + " | ".join(norm(r)) + " |")
    return "\n".join(lines)


def df_to_markdown(df) -> str:
    """pandas DataFrame -> markdown (columns become the header row)."""
    import pandas as pd

    df = df.where(pd.notna(df), "")
    header = [_esc(c) for c in df.columns]
    body = df.astype(object).values.tolist()
    return rows_to_markdown([header, *body])


_SEP_RE = re.compile(r"\|\s*:?-{3,}")


def looks_like_markdown_table(text: str) -> bool:
    """True if `text` looks like a markdown table (has a header separator row)."""
    s = (text or "").strip()
    if not s.startswith("|") or "\n" not in s:
        return False
    return any(_SEP_RE.search(line) for line in s.splitlines())
