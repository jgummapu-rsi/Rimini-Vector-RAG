"""Chunk-set diffing: what a re-ingest actually added and removed.

Re-ingesting a document replaces its chunks wholesale (see
`MetadataStore.replace_document_chunks`), which is correct but opaque -- the
caller learns "42 chunks" and nothing about whether that was a typo fix or a
rewrite. This module compares the outgoing chunk set against the incoming one so
the job trace and the version history can say `+5 / -3 / 41 unchanged`, with a
preview of what moved.

**Diffing is on `content_sha256`, never on `chunk_id`.** Chunk ids are positional
(`app.shared.ids.chunk_id` = document id + zero-padded ordinal), so inserting a
single paragraph near the top of a document shifts every subsequent chunk's id
and an id-based diff would report the entire tail as changed. Worse, the pad
width is derived from the chunk count, so a document crossing 999 -> 1000 chunks
re-keys every chunk it has. `ChunkRecord.content_sha256` is a pure function of
the chunk's final text, which is the property a diff actually needs.

The counts are multiset counts, not set counts -- see `diff_chunks`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from app.shared.domain.models import ChunkRecord

PREVIEW_CHARS = 160
PREVIEW_LIMIT = 10


def diff_chunks(old: Iterable[ChunkRecord], new: Iterable[ChunkRecord]) -> dict:
    """Compare two chunk sets by content hash; return a JSON-safe delta report.

    Multiset semantics, deliberately: duplicate chunk texts are legitimate and
    common (a repeated table header, a boilerplate footer, the same heading
    breadcrumb on two sibling sections). Counting distinct hashes would report a
    document that went from three identical footers to one as "unchanged", so
    the counts come from `Counter` arithmetic instead:

        unchanged = |old & new|   (intersection: per-hash minimum)
        added     = |new - old|   (positive difference)
        removed   = |old - new|

    Those three always sum correctly: `added + unchanged == len(new)` and
    `removed + unchanged == len(old)`.

    On a first ingest `old` is empty, so everything is `added` -- which is the
    honest report, not a special case. On a `/reprocess` of unchanged bytes
    everything is `unchanged` unless the chunking config itself moved, which
    makes this a free regression check on chunker changes.
    """
    old_list = list(old)
    new_list = list(new)

    old_counts = Counter(_hash_of(r) for r in old_list)
    new_counts = Counter(_hash_of(r) for r in new_list)

    unchanged = sum((old_counts & new_counts).values())
    added_counts = new_counts - old_counts
    removed_counts = old_counts - new_counts

    added_previews, added_truncated = _previews(new_list, added_counts)
    removed_previews, removed_truncated = _previews(old_list, removed_counts)

    return {
        "added": sum(added_counts.values()),
        "removed": sum(removed_counts.values()),
        "unchanged": unchanged,
        "added_preview": added_previews,
        "removed_preview": removed_previews,
        "added_truncated": added_truncated,
        "removed_truncated": removed_truncated,
    }


def _hash_of(record: ChunkRecord) -> str:
    """A chunk's content hash, recomputed from its text when absent.

    `finalize_chunks` stamps `content_sha256` on every chunk the pipeline
    produces, and both metadata stores persist it -- but it is typed Optional,
    and chunks written before that column existed read back as None. Keying
    those on None would collapse a whole legacy document onto one bucket and
    report it as a single unchanged chunk.

    Recomputing (rather than keying on the raw text) matters because the two
    sides of a diff can disagree: the OLD chunks come from the store and may
    predate the column, while the NEW ones were just stamped. Both sides have to
    land in the same key space or a legacy document would read as a total
    rewrite on its first re-ingest. This mirrors `finalize_chunks` exactly --
    sha256 of the chunk's final text, breadcrumb prefix included.
    """
    if record.content_sha256:
        return record.content_sha256
    return hashlib.sha256((record.text or "").encode("utf-8")).hexdigest()


def _previews(records: list[ChunkRecord], counts: Counter) -> tuple[list[str], bool]:
    """Up to PREVIEW_LIMIT truncated texts for the hashes in `counts`.

    Walks `records` in order so previews read in document order, and honours the
    multiset count (a chunk added three times is previewed three times, up to
    the cap). Returns the previews plus whether the cap cut anything off.
    """
    remaining = Counter(counts)
    previews: list[str] = []
    for record in records:
        key = _hash_of(record)
        if remaining.get(key, 0) <= 0:
            continue
        remaining[key] -= 1
        if len(previews) < PREVIEW_LIMIT:
            previews.append(_truncate(record.text))
    return previews, sum(counts.values()) > len(previews)


def _truncate(text: str) -> str:
    """Collapse whitespace and cut to PREVIEW_CHARS with an ellipsis."""
    flat = " ".join((text or "").split())
    if len(flat) <= PREVIEW_CHARS:
        return flat
    return flat[:PREVIEW_CHARS].rstrip() + "..."
