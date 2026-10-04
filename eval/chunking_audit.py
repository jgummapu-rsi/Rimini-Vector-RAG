"""Read-only replay of a stored PDF with current extraction and chunking.

Run inside the worker environment: python -m eval.chunking_audit DOCUMENT_ID.
No embeddings, generation, or index writes. Layout uses the configured local model.
"""

import argparse
import json

import psycopg2
import tiktoken

from app.ingest.adapters.localfs.blob_store import LocalFsBlobStore
from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.loaders import extract_document
from app.retrieval.rag.provenance import citation_provenance
from app.retrieval.rag.query import _quote_provenance
from app.shared.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()
    cfg = Settings()
    with psycopg2.connect(cfg.postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT filename, blob_path FROM documents WHERE id=%s", (args.document_id,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit("Document not found")
        filename, path = row
        cur.execute(
            "SELECT count(*), count(*) FILTER (WHERE token_count<50) FROM chunks WHERE document_id=%s",
            (args.document_id,),
        )
        before, before_tiny = cur.fetchone()
    if args.native_only:
        cfg = cfg.model_copy(update={"layout_enabled": False})

    class NoVision:
        def vision(self, *args, **kwargs):
            raise RuntimeError("This read-only audit does not call vision models")

    elements, _ = extract_document(
        filename, LocalFsBlobStore(cfg.blob_dir).get(path), NoVision(), cfg
    )
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_elements(
        elements,
        ChunkSpec.auto(8191),
        embed_max=8191,
        count=lambda text: len(encoding.encode(text, disallowed_special=())),
    )
    mapped, quotes = 0, 0
    for chunk in chunks:
        provenance = citation_provenance(chunk.meta, "pdf")
        for span in chunk.meta.get("source_spans", []):
            if span.get("precision") == "word" and len(span.get("text", "")) > 3:
                text = span["text"]
                # Repeated single words are intentionally ambiguous.
                if sum(region.get("text") == text for region in provenance["regions"]) == 1:
                    quotes += 1
                    mapped += _quote_provenance(provenance, text)["selection_status"] == "quote"
    print(
        json.dumps(
            {
                "filename": filename,
                "published_chunks": before,
                "published_under_50": before_tiny,
                "replay_chunks": len(chunks),
                "replay_under_50": sum(c.token_count < 50 for c in chunks),
                "unique_word_quote_probes": quotes,
                "mapped_word_quote_probes": mapped,
                "chunks": [
                    {
                        "ordinal": c.ordinal,
                        "tokens": c.token_count,
                        "pages": c.meta.get("pages"),
                        "section": c.meta.get("section_path"),
                        **({"text": c.text} if args.text else {}),
                    }
                    for c in chunks
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
