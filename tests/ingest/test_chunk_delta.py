"""Unit tests for the re-ingest chunk diff (app.ingest.pipeline.delta).

This is the function that answers "what did this new version actually change?",
and it has two properties that are easy to get wrong and invisible when they
are:

1. It must diff on CONTENT, not on chunk id. Chunk ids are positional
   (`app.shared.ids.chunk_id` = document id + zero-padded ordinal), so inserting
   a paragraph near the top of a document shifts every id after it and an
   id-based diff would report the whole tail as rewritten.

2. It must use MULTISET arithmetic, not sets. Duplicate chunk texts are normal
   -- a repeated table header, a boilerplate footer, the same heading breadcrumb
   on sibling sections. A set-based diff reports a document that went from three
   identical footers to one as completely unchanged.

The previews are bounded on purpose: the delta is persisted as JSON on a
job_events row and a document_versions row, so an unbounded preview list would
embed a rewritten document into the trace twice over.
"""

from __future__ import annotations

import hashlib

from app.ingest.pipeline.delta import PREVIEW_CHARS, PREVIEW_LIMIT, diff_chunks
from app.shared.domain.models import ChunkRecord


def _chunk(text: str, ordinal: int = 0, *, hashed: bool = True) -> ChunkRecord:
    """A ChunkRecord shaped like one the pipeline would hand to the store."""
    return ChunkRecord(
        ordinal=ordinal,
        modality="text",
        extractor="test",
        route_reason="test",
        token_count=len(text.split()),
        text=text,
        content_sha256=(hashlib.sha256(text.encode("utf-8")).hexdigest() if hashed else None),
    )


def _chunks(*texts: str) -> list[ChunkRecord]:
    return [_chunk(t, i) for i, t in enumerate(texts)]


def test_first_ingest_reports_everything_as_added():
    delta = diff_chunks([], _chunks("alpha", "beta", "gamma"))

    assert delta["added"] == 3
    assert delta["removed"] == 0
    assert delta["unchanged"] == 0
    assert delta["added_preview"] == ["alpha", "beta", "gamma"]
    assert delta["removed_preview"] == []


def test_identical_chunk_sets_are_all_unchanged():
    old = _chunks("alpha", "beta")
    new = _chunks("alpha", "beta")

    delta = diff_chunks(old, new)

    assert delta == {
        "added": 0,
        "removed": 0,
        "unchanged": 2,
        "added_preview": [],
        "removed_preview": [],
        "added_truncated": False,
        "removed_truncated": False,
    }


def test_add_and_remove_are_reported_separately_with_previews():
    old = _chunks("keep me", "delete me", "also keep")
    new = _chunks("keep me", "also keep", "brand new")

    delta = diff_chunks(old, new)

    assert delta["added"] == 1
    assert delta["removed"] == 1
    assert delta["unchanged"] == 2
    assert delta["added_preview"] == ["brand new"]
    assert delta["removed_preview"] == ["delete me"]


def test_insertion_at_the_top_does_not_look_like_a_rewrite():
    """The whole reason the diff is on content and not on chunk id.

    Every chunk after the insertion point gets a different ordinal, and
    therefore a different `chunk_id`, while its text is untouched. Diffing on
    id would call this 3 removals + 4 additions.
    """
    old = _chunks("one", "two", "three")
    new = _chunks("zero", "one", "two", "three")

    delta = diff_chunks(old, new)

    assert delta["added"] == 1
    assert delta["removed"] == 0
    assert delta["unchanged"] == 3
    assert delta["added_preview"] == ["zero"]


def test_duplicate_chunk_texts_are_counted_as_a_multiset():
    """Three identical footers becoming one is a removal of two, not a no-op."""
    old = _chunks("footer", "footer", "footer", "body")
    new = _chunks("footer", "body")

    delta = diff_chunks(old, new)

    assert delta["removed"] == 2
    assert delta["unchanged"] == 2
    assert delta["added"] == 0
    assert delta["removed_preview"] == ["footer", "footer"]


def test_duplicates_added_are_also_counted_per_occurrence():
    old = _chunks("header", "body")
    new = _chunks("header", "header", "header", "body")

    delta = diff_chunks(old, new)

    assert delta["added"] == 2
    assert delta["unchanged"] == 2
    assert delta["added_preview"] == ["header", "header"]


def test_counts_reconcile_with_both_chunk_set_sizes():
    """added + unchanged == len(new), and removed + unchanged == len(old).

    The invariant that makes the report trustworthy: nothing is double-counted
    or dropped, whatever the overlap looks like.
    """
    old = _chunks("a", "b", "b", "c", "d")
    new = _chunks("b", "c", "c", "e")

    delta = diff_chunks(old, new)

    assert delta["added"] + delta["unchanged"] == len(new)
    assert delta["removed"] + delta["unchanged"] == len(old)


def test_emptied_document_reports_every_chunk_removed():
    """A replacement that parses to nothing (corrupt/empty upload)."""
    delta = diff_chunks(_chunks("alpha", "beta"), [])

    assert delta["added"] == 0
    assert delta["removed"] == 2
    assert delta["unchanged"] == 0
    assert delta["removed_preview"] == ["alpha", "beta"]


def test_previews_are_capped_and_flag_the_truncation():
    new = _chunks(*[f"chunk number {i}" for i in range(PREVIEW_LIMIT + 5)])

    delta = diff_chunks([], new)

    assert delta["added"] == PREVIEW_LIMIT + 5
    assert len(delta["added_preview"]) == PREVIEW_LIMIT
    assert delta["added_truncated"] is True
    assert delta["removed_truncated"] is False


def test_preview_text_is_truncated_and_whitespace_collapsed():
    long_text = "word " * 200
    delta = diff_chunks([], [_chunk(long_text)])

    preview = delta["added_preview"][0]
    assert preview.endswith("...")
    assert len(preview) <= PREVIEW_CHARS + 3
    assert "  " not in preview
    assert "\n" not in preview


def test_unhashed_legacy_chunks_fall_back_to_text():
    """Chunks stored before `content_sha256` existed read back as None.

    Keying them all on None would collapse a whole legacy document onto one
    bucket and report it as a single unchanged chunk.
    """
    old = [_chunk("alpha", 0, hashed=False), _chunk("beta", 1, hashed=False)]
    new = _chunks("alpha", "gamma")

    delta = diff_chunks(old, new)

    assert delta["unchanged"] == 1
    assert delta["added"] == 1
    assert delta["removed"] == 1
