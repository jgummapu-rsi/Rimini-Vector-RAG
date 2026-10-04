"""Structure-aware, sentence-safe chunking.

Principles:
- Respect natural boundaries first (paragraphs / headings / table rows), fall
  back to sentences, and only hard-split by words as a last resort.
- Never cut mid-sentence unless a single sentence exceeds the hard cap.
- Merge consecutive TEXT elements so paragraphs aren't chopped into tiny chunks,
  but let TABLE / IMAGE elements stand as their own chunks.
- Tables split by rows, repeating the header on every chunk (retrieval-friendly).
- Overlap is paragraph/sentence-aligned (carry whole trailing units), not a raw
  token slice — keeps chunk boundaries clean.
- Every chunk stays within the embedding model's real token limit (see
  `app.ingest.pipeline.tokens`) with headroom reserved for the heading-path prefix
  (see `_record`), and every chunk carries its own section heading path so a
  chunk is never separated from the label that identifies what it's about.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.tokens import EMBED_MAX_TOKENS, count_tokens
from app.shared.domain.models import ChunkRecord, Modality

log = logging.getLogger(__name__)

_PREFIX_HEADROOM = 36
_TARGET_RATIO = 0.82
_OVERLAP_RATIO = 0.11


@dataclass(frozen=True)
class ChunkSpec:
    target_tokens: int = 180
    overlap_tokens: int = 20
    max_tokens: int = 220
    min_tokens: int = 16

    @classmethod
    def auto(cls, embed_max_tokens: int, min_tokens: int = 16) -> ChunkSpec:
        """Derive a spec from an embedder's real token limit, reserving prefix
        headroom. Reproduces the hand-tuned MiniLM defaults exactly at
        embed_max_tokens=256 (-> 180/20/220), and scales up for larger models
        (bge-base 512 -> 390/43/476) so a chunk uses the model's real capacity
        instead of being needlessly fragmented at MiniLM's 256."""
        if embed_max_tokens < 4:
            raise ValueError("Embedding token limit is too small for evidence")
        hard = min(476, max(3, embed_max_tokens - _PREFIX_HEADROOM))
        min_tokens = min(min_tokens, hard - 1)
        target = max(min_tokens, round(hard * _TARGET_RATIO))
        overlap = max(1, round(target * _OVERLAP_RATIO))
        return cls(
            target_tokens=target, overlap_tokens=overlap, max_tokens=hard, min_tokens=min_tokens
        )


DEFAULT_SPEC = ChunkSpec()

Counter = Callable[[str], int]

_PARA_RE = re.compile(r"\n\s*\n")

_SENT_RE = re.compile(r"(?<=[.!?])(?<!\d[.!?])(?<!\d\d[.!?])\s+")
_TABLE_SEP_RE = re.compile(r"\|\s*:?-{3,}")
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_PROSE_BOUNDARY_RE = re.compile(r"\n\s*\n|\n(?=\s*(?:[●•▪*-]|\d+[.)])\s+)")


def _is_heading_only(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return len(lines) == 1 and bool(_HEADING_LINE_RE.match(lines[0]))


def _join_text_elements(elements: list[Element]) -> list[Element]:
    """Join compatible prose before token packing, retaining each source span.

    Source IDs and layout detections describe citations, not chunk boundaries.
    Pages, sections, columns and modalities still separate runs. Original
    element offsets are kept alongside offsets into the joined text.
    """
    output: list[Element] = []
    run: list[Element] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            output.append(run[0])
            run.clear()
            return
        text, spans, offset = [], [], 0
        separator = "\n" if run[0].extractor == "pdf_text" else "\n\n"
        for element in run:
            text.append(element.text)
            sources = element.meta.get("source_spans") or [
                {
                    "element": element.meta.get("source_element", element.order),
                    "start": 0,
                    "end": len(element.text),
                    "text": element.text,
                    **{
                        key: element.meta[key]
                        for key in ("page", "bbox", "page_width", "page_height")
                        if key in element.meta
                    },
                }
            ]
            for span in sources:
                start, end = span.get("start", 0), span.get("end", len(element.text))
                spans.append(
                    {
                        **span,
                        "start": offset + start,
                        "end": offset + end,
                        "element_start": start,
                        "element_end": end,
                        "text": element.text[start:end],
                        "precision": span.get("precision", "exact_text"),
                    }
                )
            offset += len(element.text) + len(separator)
        meta = dict(
            run[0].meta,
            source_spans=spans,
            source_elements=[element.meta.get("source_element", element.order) for element in run],
        )
        # Individual locators remain on spans; none describes the whole run.
        for key in (
            "bbox",
            "source_element",
            "layout_id",
            "layout_label",
            "layout_confidence",
            "source_line",
            "source_line_start",
            "source_line_end",
            "body_index",
            "paragraph_index",
        ):
            meta.pop(key, None)
        output.append(replace(run[0], text=separator.join(text), meta=meta))
        run.clear()

    for element in elements:
        prose = (
            element.modality == "text"
            and element.route_reason != "code_block"
            and not element.meta.get("omission")
        )
        if not prose:
            flush()
            output.append(element)
            continue
        if run:
            previous = run[-1]
            same_context = (
                previous.extractor == element.extractor
                and previous.route_reason == element.route_reason
                and all(
                    previous.meta.get(key) == element.meta.get(key)
                    for key in ("page", "sheet", "section_path", "section", "frame")
                )
            )
            if not same_context or (
                element.extractor == "pdf_text"
                and previous.meta.get("bbox")
                and element.meta.get("bbox")
                and not _adjacent_pdf_lines(previous, element)
            ):
                flush()
        run.append(element)
    flush()
    return output


def _adjacent_pdf_lines(previous: Element, current: Element) -> bool:
    if any(
        previous.meta.get(key) != current.meta.get(key)
        for key in ("page", "section_path", "section", "page_width", "page_height")
    ):
        return False
    _, top, _, bottom = previous.meta["bbox"]
    _, y0, _, y1 = current.meta["bbox"]
    height = max(bottom - top, y1 - y0)
    # A reset upwards marks the next column in the loader's reading order.
    # Indentation, centered headings and right-aligned labels on the same line
    # are normal prose structure, not reasons to create another tiny chunk.
    return height > 0 and y0 >= top - height * 0.25 and y0 - bottom <= max(36, height * 3)


def chunk_elements(
    elements: list[Element],
    spec: ChunkSpec = DEFAULT_SPEC,
    *,
    count: Counter = count_tokens,
    embed_max: int = EMBED_MAX_TOKENS,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    limit = min(spec.max_tokens, embed_max)
    if limit < 3 or spec.target_tokens <= 0:
        raise ValueError("Chunk token budget must allow evidence and special tokens")
    spec = replace(
        spec,
        max_tokens=limit,
        target_tokens=min(spec.target_tokens, limit),
        min_tokens=min(spec.min_tokens, limit - 1),
    )
    for el in _join_text_elements(elements):
        if el.meta.get("omission") and not el.text.strip():
            continue
        local_spec = _reserve_for_prefix(spec, el.meta.get("section_path"), count)
        ranges = None
        if el.route_reason == "code_block":
            parts = _hard_split(el.text, local_spec.max_tokens, count)
            ranges, offset = [], 0
            for part in parts:
                ranges.append((offset, offset + len(part)))
                offset += len(part)
        elif el.modality == Modality.TABLE.value:
            parts = _chunk_table(el.text, local_spec, count)
        else:
            ranges = _prose_ranges(el.text, local_spec, count)
            parts = [el.text[start:end] for start, end in ranges]
        for index, part in enumerate(parts or [el.text]):
            if el.modality == "text" and el.route_reason != "code_block" and _is_heading_only(part):
                continue
            if el.meta.get("section_source") == "pdf_visible_heading" and part.strip() in (
                el.meta.get("section_path") or ""
            ).split(" > "):
                continue  # the heading is carried by the following section's prefix
            records.append(
                _record(
                    part,
                    el,
                    len(records),
                    _page_range([el]),
                    count,
                    limit,
                    source_range=ranges[index] if ranges else None,
                )
            )
    final = []
    for record in records:
        prefix = _bounded_heading(record.meta.get("section_path"), limit, count)
        body = record.text
        if prefix and body.startswith(prefix + "\n\n"):
            body = body[len(prefix) + 2 :]
        body_budget = limit - count(prefix + "\n\n") if prefix else limit
        offset = 0
        for fragment in _hard_split(body, max(3, body_budget), count):
            part = f"{prefix}\n\n{fragment}" if prefix else fragment
            meta = dict(record.meta)
            if part != record.text:
                meta["continuation"] = True
            spans = meta.get("source_spans", [])
            source_range = meta.pop("_source_range", None)
            if source_range and fragment != body:
                start = source_range[0] + offset
                meta["source_spans"] = _clip_source_spans(spans, start, start + len(fragment))
            offset += len(fragment)
            if record.modality == "table" and fragment != body:
                raise ValueError("Final table split would separate cell identity")
            final.append(
                replace(record, ordinal=len(final), text=part, token_count=count(part), meta=meta)
            )
    if any(record.token_count > limit for record in final):
        raise ValueError("Final chunk token budget violated")
    return final


def _prose_ranges(text: str, spec: ChunkSpec, count: Counter) -> list[tuple[int, int]]:
    """Pack source intervals, never reconstructed strings or substring guesses.

    Paragraphs and list items are preferred units; long units fall back to whole
    sentences, then token-bounded source fragments. Overlap includes only complete
    trailing units and is dropped when it would displace new evidence.
    """
    units = []

    def append(start, end):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            return
        if count(text[start:end]) <= spec.max_tokens:
            units.append((start, end))
        else:
            offset = start
            for part in _hard_split(text[start:end], spec.target_tokens, count):
                units.append((offset, offset + len(part)))
                offset += len(part)

    boundaries = [0, *(match.end() for match in _PROSE_BOUNDARY_RE.finditer(text)), len(text)]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if count(text[start:end]) <= spec.target_tokens:
            append(start, end)
        else:
            cuts = [
                start,
                *(start + match.end() for match in _SENT_RE.finditer(text[start:end])),
                end,
            ]
            for left, right in zip(cuts, cuts[1:], strict=False):
                append(left, right)
    output, current = [], []
    for unit in units:
        if current and count(text[current[0][0] : unit[1]]) > spec.target_tokens:
            output.append((current[0][0], current[-1][1]))
            tail = []
            for previous in reversed(current):
                if count(text[previous[0] : current[-1][1]]) > spec.overlap_tokens:
                    break
                tail.insert(0, previous)
            current = tail
            while current and count(text[current[0][0] : unit[1]]) > spec.max_tokens:
                current.pop(0)
        current.append(unit)
    if current:
        last = (current[0][0], current[-1][1])
        if (
            output
            and count(text[last[0] : last[1]]) < spec.min_tokens
            and count(text[output[-1][0] : last[1]]) <= spec.max_tokens
        ):
            output[-1] = (output[-1][0], last[1])
        else:
            output.append(last)
    return output


def _bounded_heading(section_path: str | None, limit: int, count: Counter) -> str:
    if not section_path:
        return ""
    budget = max(3, limit // 3)
    if count(section_path) <= budget:
        return section_path
    nearest = section_path.rsplit(" > ", 1)[-1]
    return _hard_split(nearest, budget, count)[0]


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", row.strip().strip("|"))]


def _wide_table_records(
    header: str, rows: list[str], spec: ChunkSpec, count: Counter, caption: str = ""
) -> list[str]:
    headers = _table_cells(header)
    parsed_rows = [_table_cells(row) for row in rows]
    if any(len(cells) != len(headers) for cells in parsed_rows):
        raise ValueError("Table row/header cardinality mismatch")
    # Long prose is sometimes extracted into a header or the first table cell.
    # Keep it as evidence once, rather than repeating it in every cell's label.
    if any(
        count(
            (caption + "\n" if caption else "")
            + f"Row {row_index}; column {column} ({name}); key {cells[0]}: "
        )
        >= spec.max_tokens - 3
        for row_index, cells in enumerate(parsed_rows, 1)
        for column, name in enumerate(headers, 1)
    ):
        return _compact_table_records(header, parsed_rows, spec, count, caption)
    output = []
    for row_index, row in enumerate(rows, 1):
        cells = _table_cells(row)
        if len(cells) != len(headers):
            raise ValueError("Table row/header cardinality mismatch")
        for column, (name, value) in enumerate(zip(headers, cells, strict=True), 1):
            identity = f"Row {row_index}; column {column} ({name}); key {cells[0]}: "
            if caption:
                identity = caption + "\n" + identity
            if count(identity) >= spec.max_tokens - 3:
                raise ValueError(
                    "Table cell identity cannot fit the chunk budget without evidence loss"
                )
            budget = spec.max_tokens - count(identity) - 2
            if budget < 3:
                raise ValueError("Table cell identity cannot fit the chunk budget")
            for piece in _hard_split(value or "(empty)", budget, count):
                record = identity + piece
                if count(record) > spec.max_tokens:
                    raise ValueError("Table cell record exceeds the chunk budget")
                output.append(record)
    return output


def _compact_table_records(
    header: str, rows: list[list[str]], spec: ChunkSpec, count: Counter, caption: str
) -> list[str]:
    """Lossless field fragments linked by a stable table/row/column reference.

    Column labels and the caption become searchable records themselves. The
    first column's value remains the row key, stored like every other cell.
    """

    identity = json.dumps([caption, header, rows], ensure_ascii=False)
    reference = hashlib.sha256(identity.encode()).hexdigest()[:12]
    fields = []
    if caption:
        fields.append(("caption", caption))
    fields.extend(
        (f"column {column} header", name) for column, name in enumerate(_table_cells(header), 1)
    )
    fields.extend(
        (f"Row {row}; column {column}", value)
        for row, cells in enumerate(rows, 1)
        for column, value in enumerate(cells, 1)
    )
    output = []
    for label, value in fields:
        prefix = f"Table {reference}; {label}: "
        budget = spec.max_tokens - count(prefix) - 2
        if budget < 1:
            raise ValueError("Chunk budget cannot fit a compact table identity")
        # Count the complete string as token boundaries need not be additive.
        while True:
            pieces = _hard_split(value or "(empty)", budget, count)
            if all(count(prefix + piece) <= spec.max_tokens for piece in pieces):
                break
            budget -= 1
            if budget < 1:
                raise ValueError("Chunk budget cannot fit a compact table identity and evidence")
        output.extend(prefix + piece for piece in pieces)
    return output


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_RE.split(text or "") if p.strip()]


def _to_units(paras: list[str], spec: ChunkSpec, count: Counter = count_tokens) -> list[str]:
    """Split paragraphs larger than the target into sentence-groups so that no
    unit ever exceeds `target_tokens`. This keeps every emitted chunk within
    `target + overlap` <= `max_tokens`."""
    units: list[str] = []
    for p in paras:
        if count(p) <= spec.target_tokens:
            units.append(p)
        else:
            units.extend(_split_oversized(p, spec, count))
    return units


def _pack_units(units: list[str], spec: ChunkSpec, count: Counter = count_tokens) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for u in units:
        ut = count(u)
        if cur and cur_tok + ut > spec.target_tokens:
            text = "\n\n".join(cur)
            chunks.append(text)
            seed = _tail_sentences(text, spec.overlap_tokens, count)
            cur = [seed] if seed else []
            cur_tok = count(seed) if seed else 0
        cur.append(u)
        cur_tok += ut
    if cur:
        chunks.append("\n\n".join(cur))
    return _merge_small(chunks, spec, count)


def _tail_sentences(text: str, budget: int, count: Counter = count_tokens) -> str:
    """Trailing whole sentences of `text` totalling <= budget tokens (never a
    partial or oversized sentence, so overlap can't inflate the next chunk).
    Never reaches back past a heading line -- a heading belongs to the chunk
    it introduces, not to an overlap seed carried into the next one (which
    either starts a new section entirely, or continues this one where the
    heading is already visible at the top -- repeating it via overlap would
    only be redundant, never useful)."""
    if budget <= 0:
        return ""
    lines = text.splitlines()
    last_heading = max((i for i, ln in enumerate(lines) if _HEADING_LINE_RE.match(ln)), default=-1)
    if last_heading >= 0:
        text = "\n".join(lines[last_heading + 1 :])
    tail: list[str] = []
    tok = 0
    for s in reversed(_SENT_RE.split(text)):
        s = s.strip()
        if not s:
            continue
        t = count(s)
        if tok + t > budget:
            break
        tail.insert(0, s)
        tok += t
    return " ".join(tail)


def _merge_small(chunks: list[str], spec: ChunkSpec, count: Counter = count_tokens) -> list[str]:
    if len(chunks) < 2:
        return chunks
    last = chunks[-1]
    if count(last) < spec.min_tokens:
        merged = chunks[-2] + "\n\n" + last
        if count(merged) <= spec.max_tokens:
            return chunks[:-2] + [merged]
    return chunks


def _split_oversized(paragraph: str, spec: ChunkSpec, count: Counter = count_tokens) -> list[str]:
    """Split a too-big paragraph on sentence boundaries; hard-split only if a
    single sentence still exceeds the cap."""
    groups: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in _SENT_RE.split(paragraph):
        sent = sent.strip()
        if not sent:
            continue
        st = count(sent)
        if st > spec.max_tokens:
            if cur:
                groups.append(" ".join(cur))
                cur, cur_tok = [], 0
            groups.extend(_hard_split(sent, spec.target_tokens, count))
            continue
        if cur and cur_tok + st > spec.target_tokens:
            groups.append(" ".join(cur))
            cur, cur_tok = [], 0
        cur.append(sent)
        cur_tok += st
    if cur:
        groups.append(" ".join(cur))
    return groups


def _hard_split(sentence: str, target_tokens: int, count: Counter = count_tokens) -> list[str]:
    if not sentence:
        return []
    out = []
    remaining = sentence
    while remaining:
        if count(remaining) <= target_tokens:
            out.append(remaining)
            break
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            midpoint = (low + high) // 2
            if count(remaining[:midpoint]) <= target_tokens:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        if not best:
            raise ValueError("One source character exceeds the chunk budget")
        boundary = remaining.rfind(" ", 0, best)
        if boundary > best // 2 and count(remaining[: boundary + 1]) <= target_tokens:
            best = boundary + 1
        part = remaining[:best]
        if count(part) > target_tokens:
            raise ValueError("Source fragment exceeds the actual tokenizer budget")
        out.append(part)
        remaining = remaining[best:]
    return out


def _chunk_table(md: str, spec: ChunkSpec, count: Counter = count_tokens) -> list[str]:
    lines = [ln for ln in (md or "").splitlines() if ln.strip()]

    sep_idx = next((i for i, ln in enumerate(lines) if _TABLE_SEP_RE.search(ln)), None)
    if sep_idx is None or sep_idx == 0:
        return _pack_units(_to_units(_split_paragraphs(md), spec, count), spec, count) or [md]
    if count(md) <= spec.max_tokens:
        return [md]

    prefix = lines[: sep_idx + 1]
    rows = lines[sep_idx + 1 :]
    base_tok = count("\n".join(prefix))
    if any(count("\n".join(prefix + [row])) > spec.max_tokens for row in rows):
        return _wide_table_records(
            lines[sep_idx - 1], rows, spec, count, "\n".join(lines[: sep_idx - 1])
        )
    if base_tok > spec.target_tokens * 0.8:
        log.warning(
            "table caption/header alone consumes %d of the %d-token target -- "
            "row-splitting budget is nearly exhausted before any row is packed",
            base_tok,
            spec.target_tokens,
            extra={
                "event": "table_prefix_overhead",
                "base_tokens": base_tok,
                "target_tokens": spec.target_tokens,
            },
        )
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = base_tok
    overlap_row: str | None = None
    for r in rows:
        rt = count(r)

        if base_tok + rt > spec.max_tokens:
            if cur:
                chunks.append("\n".join(prefix + cur))
                cur, cur_tok = [], base_tok
            for piece in _hard_split(r, max(1, spec.target_tokens - base_tok), count):
                chunks.append("\n".join(prefix + [piece]))
            overlap_row = None
            continue
        if cur and cur_tok + rt > spec.target_tokens:
            chunks.append("\n".join(prefix + cur))
            overlap_row = cur[-1]
            cur, cur_tok = [], base_tok

            if overlap_row and base_tok + count(overlap_row) + rt <= spec.max_tokens:
                cur.append(overlap_row)
                cur_tok += count(overlap_row)
        cur.append(r)
        cur_tok += rt
    if cur:
        chunks.append("\n".join(prefix + cur))
    return chunks


def _page_range(elements) -> list[int]:
    pages = sorted({e.meta.get("page") for e in elements if e.meta.get("page")})
    return [p for p in pages if p is not None]


def _reserve_for_prefix(
    spec: ChunkSpec, section_path: str | None, count: Counter = count_tokens
) -> ChunkSpec:
    """Shrink the packing budget by the ancestors-prefix's real token cost so the
    *post-prepend* chunk (built in `_record`) stays within `spec.max_tokens`,
    instead of packing against the full budget and finding out only afterward
    that the prefix pushed it over."""
    prefix = _bounded_heading(section_path, spec.max_tokens, count)
    if not prefix:
        return spec
    reserve = count(prefix) + 2
    return ChunkSpec(
        target_tokens=max(3, spec.target_tokens - reserve),
        overlap_tokens=spec.overlap_tokens,
        max_tokens=max(3, spec.max_tokens - reserve),
        min_tokens=min(spec.min_tokens, max(1, spec.max_tokens - reserve - 1)),
    )


def _record(
    text: str,
    el: Element,
    ordinal: int,
    pages: list[int],
    count: Counter = count_tokens,
    embed_max: int = EMBED_MAX_TOKENS,
    source_range: tuple[int, int] | None = None,
) -> ChunkRecord:
    meta = dict(el.meta)
    source_spans = [dict(span) for span in meta.get("source_spans", [])]
    if source_spans and el.modality == "text":
        source_range = source_range or _find_source_range(el.text, text)
        if source_range:
            source_spans = _clip_source_spans(source_spans, *source_range)
            meta["_source_range"] = list(source_range)
        else:
            source_spans = [dict(span, precision="element_region") for span in source_spans]
    if source_spans:
        meta["source_spans"] = source_spans
        if "source_elements" in meta:
            meta["source_elements"] = list(
                dict.fromkeys(span["element"] for span in source_spans if "element" in span)
            )
            line_starts = [
                span.get("source_line_start", span.get("source_line")) for span in source_spans
            ]
            line_ends = [
                span.get("source_line_end", span.get("source_line_start", span.get("source_line")))
                for span in source_spans
            ]
            if all(isinstance(value, int) for value in line_starts + line_ends):
                meta["source_line_start"], meta["source_line_end"] = (
                    min(line_starts),
                    max(line_ends),
                )
            for key in ("body_index", "paragraph_index"):
                values = {span.get(key) for span in source_spans}
                if len(values) == 1 and None not in values:
                    meta[key] = values.pop()
    if pages:
        meta["pages"] = pages

    ancestors = _bounded_heading(meta.get("section_path"), embed_max, count)
    if ancestors:
        text = f"{ancestors}\n\n{text}"

    token_count = count(text)

    return ChunkRecord(
        ordinal=ordinal,
        modality=el.modality,
        extractor=el.extractor,
        route_reason=el.route_reason,
        token_count=token_count,
        text=text,
        meta=meta,
    )


def _clip_source_spans(spans: list[dict], start: int, end: int) -> list[dict]:
    """Keep only source regions intersecting a chunk's source character range."""
    clipped = []
    for span in spans:
        span_start = span.get("start")
        span_end = span.get("end")
        if not isinstance(span_start, int) or not isinstance(span_end, int):
            clipped.append(dict(span, precision="element_region"))
            continue
        clipped_start = max(span_start, start)
        clipped_end = min(span_end, end)
        if clipped_start >= clipped_end:
            continue
        item = dict(span, start=clipped_start, end=clipped_end)
        if "element_start" in span:
            item["element_start"] = span["element_start"] + clipped_start - span_start
            item["element_end"] = item["element_start"] + clipped_end - clipped_start
        if isinstance(span.get("text"), str):
            relative_start = clipped_start - span_start
            relative_end = clipped_end - span_start
            item["text"] = span["text"][relative_start:relative_end]
        clipped.append(item)
    return clipped


def _find_source_range(source: str, fragment: str) -> tuple[int, int] | None:
    """Locate packed text after paragraph/sentence whitespace normalization."""
    position = source.find(fragment)
    if position >= 0:
        return position, position + len(fragment)

    def normalized(value: str) -> tuple[str, list[tuple[int, int]]]:
        output = []
        offsets = []
        index = 0
        while index < len(value):
            if value[index].isspace():
                end = index + 1
                while end < len(value) and value[end].isspace():
                    end += 1
                output.append(" ")
                offsets.append((index, end))
                index = end
            else:
                output.append(value[index])
                offsets.append((index, index + 1))
                index += 1
        return "".join(output), offsets

    source_normalized, source_offsets = normalized(source)
    fragment_normalized, _ = normalized(fragment)
    fragment_normalized = fragment_normalized.strip()
    position = source_normalized.find(fragment_normalized)
    if position < 0 or not fragment_normalized:
        return None
    return source_offsets[position][0], source_offsets[position + len(fragment_normalized) - 1][1]
