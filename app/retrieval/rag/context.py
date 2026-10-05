"""Source-order evidence expansion after relevance retrieval.

Keep neighbors as independently authorized/citable chunks. The generator receives
their source order rather than guessing relationships from relevance rank.
"""

from __future__ import annotations

import re
from dataclasses import replace

from app.retrieval.rag.evidence_quality import useful_indices


def asks_for_collection(question: str) -> bool:
    return bool(
        re.search(
            r"\b(all|list|enumerate|how many|complete inventory)\b|"
            r"\b(?:identify|name|show|give)\s+(?:me\s+)?(?:every|each)\b",
            question,
            re.I,
        )
    )


def matching_roster(question: str, text: str) -> bool:
    """Match a requested collection to a table's schema, not its incidental cells."""
    if not asks_for_collection(question):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    separator = next((i for i, line in enumerate(lines) if re.match(r"^\|?\s*:?-{3,}", line)), None)
    if separator is None or separator < 1 or len(lines) - separator < 3:
        return False
    header = lines[separator - 1]
    if len(header.split()) > 16 or len(header) > 160:
        return False  # layout tables whose first row is prose, not a schema

    def words(value):
        return {word.casefold().rstrip("s") for word in re.findall(r"\w+", value) if len(word) > 2}

    ignored = {
        "the",
        "all",
        "are",
        "who",
        "what",
        "which",
        "for",
        "and",
        "list",
        "every",
        "each",
        "name",
        "use",
        "uses",
        "used",
        "when",
        "where",
        "how",
        "may",
        "can",
        "want",
        "get",
        "value",
        "type",
        "description",
        "status",
        "permitted",
        "only",
        "with",
        "their",
        "they",
    }
    # Require a collection noun in a short column label. Job titles and other
    # incidental long cells are not a list schema.
    columns = [cell.strip() for cell in header.strip("|").split("|") if cell.strip()]
    return any(
        len(cell.split()) <= 3 and bool((words(question) - ignored) & (words(cell) - ignored))
        for cell in columns
    )


def expand_evidence(container, tenant_id, retrieval, access=None):
    neighbors = container.vectors.surrounding_chunks(
        tenant_id, retrieval.chunk_ids, access=access, radius=2, limit=50
    )
    if not neighbors:
        return retrieval

    original = {
        cid: (text, score, citation)
        for cid, text, score, citation in zip(
            retrieval.chunk_ids,
            retrieval.contexts,
            retrieval.scores,
            retrieval.citations,
            strict=False,
        )
    }
    entries = dict(original)
    for hit in neighbors:
        payload = hit.payload
        citation = {
            "chunk_id": hit.chunk_id,
            "document_id": payload.get("_id"),
            **{
                key: payload.get(key)
                for key in (
                    "generation_id",
                    "version",
                    "source_type",
                    "filename",
                    "location",
                    "section_path",
                    "source_ordinal",
                    "provenance",
                )
            },
            "snippet": payload.get("content", ""),
            "score": 0.0,
        }
        if hit.chunk_id in entries:
            text, score, previous = entries[hit.chunk_id]
            entries[hit.chunk_id] = (
                text,
                score,
                dict(previous, source_ordinal=payload.get("source_ordinal")),
            )
        elif len(entries) < 50:
            entries[hit.chunk_id] = (payload.get("content", ""), 0.0, citation)
    # Preserve document/section grouping and actual source order. In particular,
    # table continuations follow their header, and duties follow their employer.
    ordered = sorted(
        entries,
        key=lambda cid: (
            entries[cid][2].get("document_id") or "",
            entries[cid][2].get("source_ordinal")
            if entries[cid][2].get("source_ordinal") is not None
            else 10**9,
            cid,
        ),
    )
    before_filter = len(ordered)
    keep = useful_indices(
        retrieval.question,
        [entries[cid][0] for cid in ordered],
        [entries[cid][2] for cid in ordered],
    )
    ordered = [ordered[i] for i in keep]
    rosters = [cid for cid in ordered if matching_roster(retrieval.question, entries[cid][0])]
    if asks_for_collection(retrieval.question) and rosters:
        # A table can enumerate dozens of items in two chunks. Keep every fetched
        # continuation ahead of incidental mentions; do not fill the input with
        # biographies when the requested evidence is a roster.
        # A roster is additional evidence, never permission to discard the
        # relevance-selected prose (which may contain rules or exceptions).
        ordered = rosters + [cid for cid in ordered if cid not in rosters]
        seen_rows = {}
        for cid in rosters:
            text, score, citation = entries[cid]
            lines = text.splitlines()
            separator = next(
                i for i, line in enumerate(lines) if re.match(r"^\|?\s*:?-{3,}", line.strip())
            )
            key = (
                citation.get("document_id"),
                citation.get("generation_id"),
                citation.get("section_path"),
                lines[separator - 1].strip(),
            )
            seen = seen_rows.setdefault(key, set())
            kept = lines[: separator + 1]
            rows = {line.strip() for line in lines[separator + 1 :] if line.strip()}
            kept.extend(line for line in lines[separator + 1 :] if line.strip() not in seen)
            seen.update(rows)
            entries[cid] = ("\n".join(kept), score, citation)
    trace = [
        *retrieval.trace,
        {
            "stage": "expand",
            "detail": f"selected {len(ordered)} source-ordered passages from {len(original)} relevance matches and their neighbors",
            "seed_count": len(original),
            "evidence_count": len(ordered),
            "roster_chunks": len(rosters),
            "coverage": "bounded_source_windows",
            "collection_requested": asks_for_collection(retrieval.question),
            "furniture_removed": before_filter - len(keep),
        },
    ]
    return replace(
        retrieval,
        contexts=[entries[cid][0] for cid in ordered],
        chunk_ids=ordered,
        scores=[entries[cid][1] for cid in ordered],
        citations=[entries[cid][2] for cid in ordered],
        trace=trace,
    )
