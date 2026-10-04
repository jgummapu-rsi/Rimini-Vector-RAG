from __future__ import annotations

import json
import logging

import ir_datasets
from psycopg2.extras import execute_values

from app.shared.adapters.pgvector.vector_store import lexical_query
from app.shared.adapters.postgres.db import transaction
from scripts.disposable_storage import evaluation_settings

log = logging.getLogger(__name__)


def main() -> None:
    dataset = ir_datasets.load("beir/scifact/test")
    corpus = [(doc.doc_id, doc.title + "\n" + doc.text) for doc in dataset.docs_iter()]
    queries = {query.query_id: query.text for query in dataset.queries_iter()}
    relevant = {}
    for judgment in dataset.qrels_iter():
        if judgment.relevance > 0:
            relevant.setdefault(judgment.query_id, set()).add(judgment.doc_id)
    with evaluation_settings() as cfg:
        with transaction(cfg.postgres_dsn) as cur:
            cur.execute("CREATE TABLE lexical_documents(id TEXT PRIMARY KEY,tsv TSVECTOR NOT NULL)")
            execute_values(
                cur,
                "INSERT INTO lexical_documents SELECT id,to_tsvector('english',body) FROM (VALUES %s) v(id,body)",
                corpus,
            )
            cur.execute("CREATE INDEX ON lexical_documents USING gin(tsv)")
        metrics = []
        for name, transform in (("websearch", lambda text: text), ("recall_or", lexical_query)):
            hits = total = zero = 0
            with transaction(cfg.postgres_dsn) as cur:
                for query_id, gold in relevant.items():
                    cur.execute(
                        "SELECT id FROM lexical_documents,websearch_to_tsquery('english',%s) q "
                        "WHERE tsv @@ q ORDER BY ts_rank_cd(tsv,q) DESC,id LIMIT 50",
                        (transform(queries[query_id]),),
                    )
                    ranked = {row["id"] for row in cur.fetchall()}
                    hits += len(ranked & gold)
                    total += len(gold)
                    zero += not ranked
            metrics.append(
                {
                    "policy": name,
                    "corpus_documents": len(corpus),
                    "queries": len(relevant),
                    "recovered_at_50": hits,
                    "relevant": total,
                    "zero_results": zero,
                }
            )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
