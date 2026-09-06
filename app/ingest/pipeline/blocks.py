"""Split markdown-ish text into structured Elements.

Used for BOTH markdown files and LLM/OCR output (a vision transcription is often
prose + embedded GFM tables). Splits into:
  - GFM tables  -> TABLE elements (nearest heading prepended as a caption)
  - fenced code -> atomic TEXT (never split)
  - everything else -> TEXT (modality configurable: text for pages, image for figures)

This is what makes scanned tables first-class: a table inside an OCR result is
isolated so the chunker keeps it intact / row-splits it with the header repeated.

A level-aware heading STACK is tracked while walking the document (not just the
nearest heading line), and prose is flushed on every heading change, so every
Element's boundary aligns with a section boundary. Each Element carries both
`meta["section"]` (nearest heading only, unchanged -- existing consumers rely on
this) and `meta["section_path"]` (the full "Doc Title > H2 > H3" breadcrumb, new).
Without this, two different sections that happen to share the same low-level
heading text (e.g. "Common Failure Pattern" repeated under 10 different topics in
the same document -- a real case in our own corpus) are indistinguishable from
each other once chunked; the full path disambiguates them.
"""
from __future__ import annotations

import re

from app.shared.domain.models import Modality
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.tables import rows_to_markdown

_SEP_RE = re.compile(r"\|\s*:?-{3,}")            # table header/body separator
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+\S")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Cap how many levels deep an ancestor path gets before it's truncated (keep the
# document title + the innermost levels, drop the noisy middle). The token
# budget for this prefix is already accounted for dynamically wherever it's
# used (chunker._reserve_for_prefix measures the real cost, doesn't guess), so
# this cap is about readability/signal-to-noise for deeply nested documents,
# not safety.
_MAX_PATH_DEPTH = 4


def _heading_level(line: str) -> int:
    m = _HEADING_RE.match(line)
    return len(m.group(1)) if m else 0


def _tab_columns(line: str) -> int | None:
    """Column count if `line` looks like a tab-separated row, else None."""
    if "\t" not in line or not line.strip():
        return None
    return len(line.split("\t"))


def split_blocks(
    text: str,
    *,
    text_extractor: str,
    table_extractor: str,
    text_reason: str,
    table_reason: str,
    text_modality: str = Modality.TEXT.value,
    meta: dict | None = None,
) -> list[Element]:
    base_meta = meta or {}
    lines = (text or "").splitlines()
    els: list[Element] = []
    order = 0
    heading_stack: list[tuple[int, str]] = []  # [(level, heading_text), ...], outermost first
    prose: list[str] = []
    i, n = 0, len(lines)

    def path() -> str:
        levels = heading_stack
        if len(levels) > _MAX_PATH_DEPTH:
            # keep the document title (outermost) + the innermost levels
            levels = [levels[0]] + levels[-(_MAX_PATH_DEPTH - 1):]
        return " > ".join(h for _, h in levels)

    def flush_prose() -> None:
        nonlocal order
        t = "\n".join(prose).strip()
        prose.clear()
        if t:
            emeta = dict(base_meta)
            if heading_stack:
                emeta["section_path"] = path()
            els.append(Element(t, text_modality, text_extractor, text_reason,
                               order, emeta))
            order += 1

    while i < n:
        line = lines[i]

        # fenced code block -> atomic
        m = _FENCE_RE.match(line)
        if m:
            fence = m.group(1)
            block = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i])
                i += 1
            flush_prose()
            els.append(Element("\n".join(block), text_modality, text_extractor,
                               "code_block", order, dict(base_meta)))
            order += 1
            continue

        # GFM table: a "| ... |" row immediately followed by a separator row
        if "|" in line and i + 1 < n and _SEP_RE.search(lines[i + 1]):
            flush_prose()
            table = [line, lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                table.append(lines[i])
                i += 1
            md = "\n".join(table)
            tmeta = dict(base_meta)
            if heading_stack:
                nearest = heading_stack[-1][1]
                md = f"{nearest}\n{md}"          # caption for retrieval context (unchanged)
                tmeta["section"] = nearest
                tmeta["section_path"] = path()
            els.append(Element(md, Modality.TABLE.value, table_extractor,
                               table_reason, order, tmeta))
            order += 1
            continue

        # Tab-separated table (e.g. pasted spreadsheet data, or an alternate
        # vision-transcription format): 2+ consecutive lines with the same
        # tab-delimited column count and at least 2 columns. Converted to GFM
        # so every downstream consumer (chunker's header-repeat-on-split,
        # section captions, ancestors-prefix) works unchanged regardless of
        # the source format -- nothing downstream needs to know tabs exist.
        cols = _tab_columns(line)
        if cols and cols >= 2 and i + 1 < n and _tab_columns(lines[i + 1]) == cols:
            flush_prose()
            tab_lines = [line]
            i += 1
            while i < n and _tab_columns(lines[i]) == cols:
                tab_lines.append(lines[i])
                i += 1
            rows = [ln.split("\t") for ln in tab_lines]
            md = rows_to_markdown(rows)
            tmeta = dict(base_meta)
            if heading_stack:
                nearest = heading_stack[-1][1]
                md = f"{nearest}\n{md}"
                tmeta["section"] = nearest
                tmeta["section_path"] = path()
            els.append(Element(md, Modality.TABLE.value, table_extractor,
                               table_reason, order, tmeta))
            order += 1
            continue

        level = _heading_level(line)
        if level:
            flush_prose()  # attribute everything so far to the OLD section path
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, line.strip()))
        prose.append(line)
        i += 1

    flush_prose()
    return _drop_redundant_heading_prose(els)


def _is_heading_only(text: str) -> bool:
    body = [ln for ln in text.strip().splitlines() if ln.strip()]
    return len(body) == 1 and bool(_HEADING_RE.match(body[0]))


def _drop_redundant_heading_prose(els: list[Element]) -> list[Element]:
    """Drop a heading-only TEXT element when the next element is a TABLE already
    carrying that heading as its caption (avoids tiny duplicate chunks)."""
    out: list[Element] = []
    for idx, e in enumerate(els):
        nxt = els[idx + 1] if idx + 1 < len(els) else None
        if (e.modality == Modality.TEXT.value and _is_heading_only(e.text)
                and nxt is not None and nxt.modality == Modality.TABLE.value
                and nxt.meta.get("section") == e.text.strip()):
            continue
        out.append(e)
    return out
