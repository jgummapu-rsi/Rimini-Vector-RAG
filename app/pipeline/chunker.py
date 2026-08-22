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
  `app.pipeline.tokens`) with headroom reserved for the heading-path prefix
  (see `_record`), and every chunk carries its own section heading path so a
  chunk is never separated from the label that identifies what it's about.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from app.domain.models import ChunkRecord, Modality
from app.pipeline.elements import Element
from app.pipeline.tokens import EMBED_MAX_TOKENS, count_tokens

log = logging.getLogger("pipeline.chunker")

# Headroom reserved under the embedder's hard limit for the section-path prefix
# `_record` prepends plus the tokenizer's special tokens ([CLS]/[SEP]). Roughly
# constant (a function of heading depth, not model size), so it's an absolute
# reserve, not a fraction. 36 under 256 is what the hand-tuned MiniLM defaults
# used; ChunkSpec.auto reproduces those defaults exactly at max_tokens=256.
_PREFIX_HEADROOM = 36
_TARGET_RATIO = 0.82   # target as a fraction of the chunk max (220*0.82 -> 180)
_OVERLAP_RATIO = 0.11  # overlap as a fraction of target (180*0.11 -> 20)


@dataclass(frozen=True)
class ChunkSpec:
    # Sized against the embedding model's REAL token limit (see
    # app.ports.embedder.Embedder.max_tokens / app.pipeline.tokens), not a
    # generic estimate -- with headroom reserved for the section-path prefix
    # `_record` prepends. Previously targeted 512/max 768 tokens as measured by
    # a generic BPE tokenizer that doesn't match the embedder's real WordPiece
    # tokenizer; those chunks were routinely 2x the model's actual limit and got
    # silently truncated before ever being embedded. Defaults are the MiniLM
    # (256-token) sizing; use ChunkSpec.auto(embedder.max_tokens) to size for any
    # embedder (e.g. bge-base's 512 -> ~390/476).
    target_tokens: int = 180   # aim for chunks around this size
    overlap_tokens: int = 20   # sentence-aligned carry-over between text chunks
    max_tokens: int = 220      # hard cap -- leaves ~36 tokens headroom under the embedder limit
    min_tokens: int = 16       # merge a tiny trailing chunk into the previous

    @classmethod
    def auto(cls, embed_max_tokens: int, min_tokens: int = 16) -> "ChunkSpec":
        """Derive a spec from an embedder's real token limit, reserving prefix
        headroom. Reproduces the hand-tuned MiniLM defaults exactly at
        embed_max_tokens=256 (-> 180/20/220), and scales up for larger models
        (bge-base 512 -> 390/43/476) so a chunk uses the model's real capacity
        instead of being needlessly fragmented at MiniLM's 256."""
        hard = max(min_tokens + 1, embed_max_tokens - _PREFIX_HEADROOM)
        target = max(min_tokens, round(hard * _TARGET_RATIO))
        overlap = max(1, round(target * _OVERLAP_RATIO))
        return cls(target_tokens=target, overlap_tokens=overlap,
                   max_tokens=hard, min_tokens=min_tokens)


DEFAULT_SPEC = ChunkSpec()

# Type of the token-counter the chunker sizes against. Defaults to the MiniLM
# module counter; the runner passes the active embedder's own count_tokens so
# sizing is measured in the tokenizer that will actually embed the chunk.
Counter = Callable[[str], int]

_PARA_RE = re.compile(r"\n\s*\n")
# Split after sentence-ending punctuation, EXCEPT when that punctuation is
# actually a numbered-list marker ("1." / "12.") rather than a sentence end --
# without this exclusion, a numbered list ("1. Do X. 2. Do Y.") gets shredded
# at every list marker's period, tearing items away from their own list (a
# real, observed failure). Trade-off: a sentence that happens to end in a bare
# 1-2 digit number ("The result was 42.") won't split here either -- rare
# enough, and far less damaging than the list-shredding it prevents.
_SENT_RE = re.compile(r"(?<=[.!?])(?<!\d[.!?])(?<!\d\d[.!?])\s+")
_TABLE_SEP_RE = re.compile(r"\|\s*:?-{3,}")   # markdown table header/body separator
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def _is_heading_only(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return len(lines) == 1 and bool(_HEADING_LINE_RE.match(lines[0]))


def chunk_elements(
    elements: list[Element],
    spec: ChunkSpec = DEFAULT_SPEC,
    *,
    count: Counter = count_tokens,
    embed_max: int = EMBED_MAX_TOKENS,
) -> list[ChunkRecord]:
    """Chunk `elements` under `spec`, measuring every size decision with `count`
    (the active embedder's tokenizer -- see app.ports.embedder). `embed_max` is
    that embedder's hard limit, used only for the over-limit safety warning.
    Both default to the MiniLM counter so callers that predate the multi-embedder
    path (and the unit tests) keep their exact previous behavior."""
    records: list[ChunkRecord] = []
    ordinal = 0
    text_buf: list[tuple[str, Element]] = []  # consecutive TEXT paragraphs

    def flush_text() -> None:
        nonlocal ordinal
        if not text_buf:
            return
        base = text_buf[0][1]
        pages = _page_range(e for _, e in text_buf)
        # text_buf is guaranteed single-section (flush_text is called on every
        # section-path change above), so one reservation covers the whole buffer.
        local_spec = _reserve_for_prefix(spec, base.meta.get("section_path"), count)
        for part in _pack_units(
            _to_units([p for p, _ in text_buf], local_spec, count), local_spec, count
        ):
            if _is_heading_only(part):
                # A heading with no body of its own is a pure container for its
                # subsections (e.g. "## Core Layers" immediately followed by
                # "### Presentation Layer" with nothing of its own in between).
                # Its text already appears in every child section's own
                # ancestors-prefix (`_ancestors_prefix`), so emitting it as a
                # separate, content-free chunk adds a real vector with zero
                # retrieval value -- pure index noise, not lost information.
                continue
            records.append(_record(part, base, ordinal, pages, count, embed_max))
            ordinal += 1
        text_buf.clear()

    last_section_path = None
    for el in elements:
        if el.modality == Modality.TEXT.value:
            # A section-path change also breaks the run, not just a non-text
            # element -- otherwise `flush_text` below only keeps text_buf[0]'s
            # metadata for every chunk it packs, silently dropping every other
            # section's heading path once split_blocks flushes one Element per
            # section (blocks.py). Without this, only the first of several
            # consecutive sections ever gets its heading path attached.
            this_section = el.meta.get("section_path")
            if text_buf and this_section != last_section_path:
                flush_text()
            last_section_path = this_section
            for para in _split_paragraphs(el.text):
                text_buf.append((para, el))
            continue

        flush_text()  # a non-text element breaks the text run
        last_section_path = None

        local_spec = _reserve_for_prefix(spec, el.meta.get("section_path"), count)
        if el.modality == Modality.TABLE.value:
            parts = _chunk_table(el.text, local_spec, count)
        else:  # IMAGE (caption / OCR of a figure)
            parts = _pack_units(
                _to_units(_split_paragraphs(el.text), local_spec, count), local_spec, count)
        for part in parts or [el.text]:
            records.append(_record(part, el, ordinal, _page_range([el]), count, embed_max))
            ordinal += 1

    flush_text()
    return records


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
    last_heading = max(
        (i for i, ln in enumerate(lines) if _HEADING_LINE_RE.match(ln)), default=-1
    )
    if last_heading >= 0:
        text = "\n".join(lines[last_heading + 1:])
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
    words = sentence.split()
    out: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for w in words:
        wt = count(w + " ")
        if cur and cur_tok + wt > target_tokens:
            out.append(" ".join(cur))
            cur, cur_tok = [], 0
        cur.append(w)
        cur_tok += wt
    if cur:
        out.append(" ".join(cur))
    return out


def _chunk_table(md: str, spec: ChunkSpec, count: Counter = count_tokens) -> list[str]:
    lines = [ln for ln in (md or "").splitlines() if ln.strip()]
    # locate the header/body separator row; the header is the line above it.
    sep_idx = next((i for i, ln in enumerate(lines) if _TABLE_SEP_RE.search(ln)), None)
    if sep_idx is None or sep_idx == 0:
        # not a real markdown table -> treat as text
        return _pack_units(_to_units(_split_paragraphs(md), spec, count), spec, count) or [md]
    if count(md) <= spec.max_tokens:
        return [md]

    # anything before the header row (e.g. a section-heading caption) is repeated
    # on every chunk, along with the header + separator, so each chunk is readable.
    prefix = lines[: sep_idx + 1]        # caption... + header + separator
    rows = lines[sep_idx + 1:]
    base_tok = count("\n".join(prefix))
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = base_tok
    overlap_row: str | None = None  # last row of the previous chunk, carried forward
    for r in rows:
        rt = count(r)
        # a single row that alone would exceed the hard cap: flush, then
        # hard-split that row so no chunk ever exceeds max_tokens.
        if base_tok + rt > spec.max_tokens:
            if cur:
                chunks.append("\n".join(prefix + cur))
                cur, cur_tok = [], base_tok
            for piece in _hard_split(r, spec.target_tokens - base_tok, count):
                chunks.append("\n".join(prefix + [piece]))
            overlap_row = None  # a hard-split fragment isn't a clean overlap seed
            continue
        if cur and cur_tok + rt > spec.target_tokens:
            chunks.append("\n".join(prefix + cur))
            overlap_row = cur[-1]
            cur, cur_tok = [], base_tok
            # Carry the previous chunk's last row forward too (same idea as the
            # sentence-aligned overlap text chunks get) -- a row sitting right at
            # a split boundary is then findable from either neighboring chunk.
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


def _ancestors_prefix(section_path: str | None) -> str:
    """The ANCESTOR path only (full section_path minus its last segment) -- the
    last segment (the section's own nearest heading) is already visible as a
    literal line at the top of the chunk text itself (markdown headings stay in
    prose text; table captions already carry the nearest heading). This adds
    exactly the context that was otherwise missing -- the document title and any
    parent sections -- without duplicating what's already there."""
    if section_path and " > " in section_path:
        return section_path.rsplit(" > ", 1)[0]
    return ""


def _reserve_for_prefix(spec: ChunkSpec, section_path: str | None,
                        count: Counter = count_tokens) -> ChunkSpec:
    """Shrink the packing budget by the ancestors-prefix's real token cost so the
    *post-prepend* chunk (built in `_record`) stays within `spec.max_tokens`,
    instead of packing against the full budget and finding out only afterward
    that the prefix pushed it over."""
    prefix = _ancestors_prefix(section_path)
    if not prefix:
        return spec
    reserve = count(prefix) + 2  # +2 for the blank-line separator
    return ChunkSpec(
        target_tokens=max(spec.min_tokens, spec.target_tokens - reserve),
        overlap_tokens=spec.overlap_tokens,
        max_tokens=max(spec.min_tokens, spec.max_tokens - reserve),
        min_tokens=spec.min_tokens,
    )


def _record(text: str, el: Element, ordinal: int, pages: list[int],
            count: Counter = count_tokens, embed_max: int = EMBED_MAX_TOKENS) -> ChunkRecord:
    meta = dict(el.meta)
    if pages:
        meta["pages"] = pages

    # Without this, a chunk whose packing boundary lands after its own heading
    # line is indistinguishable from any other section that happens to share
    # similar body text (the failure mode this fixes -- see
    # KHUB_COMPARISON_REPORT.md). Budget for this prefix is already reserved
    # before packing (`_reserve_for_prefix`), so this should never push a chunk
    # over the embedder's real limit -- the warning below is a safety net, not
    # the primary control.
    ancestors = _ancestors_prefix(meta.get("section_path"))
    if ancestors:
        text = f"{ancestors}\n\n{text}"

    token_count = count(text)
    if token_count > embed_max:
        log.warning(
            "chunk exceeds the embedding model's real token limit -- dense search "
            "will only see the first %d tokens of this %d-token chunk",
            embed_max, token_count,
            extra={"event": "chunk_over_embed_limit", "ordinal": ordinal,
                   "token_count": token_count, "embed_max_tokens": embed_max},
        )

    return ChunkRecord(
        ordinal=ordinal,
        modality=el.modality,
        extractor=el.extractor,
        route_reason=el.route_reason,
        token_count=token_count,
        text=text,
        meta=meta,
    )
