"""Change the visibility of already-ingested documents.

Run:  python -m scripts.set_visibility <tenant_id> <private|tenant> [document_id ...]
      python -m scripts.set_visibility <tenant_id> tenant --all
      python -m scripts.set_visibility <tenant_id> tenant --all --dry-run

Why this exists
---------------
`visibility` is stored TWICE on purpose: authoritatively on `documents`, and
denormalised into each vector record so the retrieval path can apply the ACL
without a join back to the metadata store (see `app.ingest.pipeline.runner._stage_upsert`,
which copies `doc.visibility` into every point's payload). Updating only the
`documents` row therefore changes nothing about what retrieval returns -- the
vector store is still filtering on its own stale copy.

This keeps the two in step. The alternative is `POST /documents/{id}/reprocess`,
which reaches the same end state by re-running the whole pipeline; that also
re-embeds every chunk and re-runs metadata extraction, so it costs real time and
gateway calls to change one field.

This is an admin utility standing in for the "no admin API" gap in CLAUDE.md's
roadmap, not part of the request path.
"""
from __future__ import annotations

import json
import sys

from app.shared.config import settings
from app.shared.domain.models import Visibility

_ALLOWED = (Visibility.PRIVATE.value, Visibility.TENANT.value)


def _sqlite_localfile(tenant_id: str, visibility: str, doc_ids: list[str],
                      dry_run: bool) -> int:
    from app.shared.adapters.sqlite.db import connect

    conn = connect(settings.sqlite_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if doc_ids:
            marks = ",".join("?" * len(doc_ids))
            rows = conn.execute(
                f"SELECT id, filename, visibility FROM documents "
                f"WHERE tenant_id=? AND id IN ({marks})",
                (tenant_id, *doc_ids),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, filename, visibility FROM documents WHERE tenant_id=?",
                (tenant_id,),
            ).fetchall()

        changed = 0
        for r in rows:
            if r["visibility"] == visibility:
                print(f"  = {r['filename'][:44]:<44} already {visibility}")
                continue
            print(f"  ~ {r['filename'][:44]:<44} {r['visibility']} -> {visibility}")
            changed += 1
            if dry_run:
                continue

            conn.execute("UPDATE documents SET visibility=? WHERE tenant_id=? AND id=?",
                         (visibility, tenant_id, r["id"]))
            # ...and the denormalised copy the retrieval path actually reads.
            vrow = conn.execute(
                "SELECT record FROM vector_documents WHERE _id=? AND tenant_id=?",
                (r["id"], tenant_id),
            ).fetchone()
            if vrow is None:
                print(f"    (no vector record yet -- it will pick up the new "
                      f"visibility when the job runs)")
                continue
            record = json.loads(vrow["record"])
            record["visibility"] = visibility
            conn.execute(
                "UPDATE vector_documents SET record=? WHERE _id=? AND tenant_id=?",
                (json.dumps(record), r["id"], tenant_id),
            )
        conn.commit()
        return changed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _postgres_pgvector(tenant_id: str, visibility: str, doc_ids: list[str],
                       dry_run: bool) -> int:
    from app.shared.adapters.postgres.db import transaction

    with transaction(settings.postgres_dsn) as cur:
        if doc_ids:
            cur.execute("SELECT id, filename, visibility FROM documents "
                        "WHERE tenant_id=%s AND id = ANY(%s)", (tenant_id, doc_ids))
        else:
            cur.execute("SELECT id, filename, visibility FROM documents "
                        "WHERE tenant_id=%s", (tenant_id,))
        rows = cur.fetchall()

        changed = 0
        for r in rows:
            if r["visibility"] == visibility:
                print(f"  = {r['filename'][:44]:<44} already {visibility}")
                continue
            print(f"  ~ {r['filename'][:44]:<44} {r['visibility']} -> {visibility}")
            changed += 1
            if dry_run:
                continue
            cur.execute("UPDATE documents SET visibility=%s WHERE tenant_id=%s AND id=%s",
                        (visibility, tenant_id, r["id"]))
            # jsonb_set rewrites just the one key in every chunk's payload.
            cur.execute(
                "UPDATE vector_chunks SET payload = jsonb_set(payload, '{visibility}', "
                "to_jsonb(%s::text)), updated_at=now() "
                "WHERE tenant_id=%s AND document_id=%s",
                (visibility, tenant_id, r["id"]),
            )
        return changed


def main() -> None:
    args = [a for a in sys.argv[1:]]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")
    take_all = "--all" in args
    if take_all:
        args.remove("--all")

    if len(args) < 2:
        print(__doc__)
        raise SystemExit(2)

    tenant_id, visibility, doc_ids = args[0], args[1], args[2:]
    if visibility not in _ALLOWED:
        raise SystemExit(f"visibility must be one of {list(_ALLOWED)}, got {visibility!r}")
    if not doc_ids and not take_all:
        raise SystemExit("pass document ids, or --all to change every document "
                         "in the tenant")

    backend = f"{settings.metadata_backend}/{settings.vector_backend}"
    print(f"tenant   : {tenant_id}")
    print(f"backend  : {backend}")
    print(f"target   : visibility={visibility}")
    print(f"documents: {'ALL in tenant' if take_all else ', '.join(doc_ids)}")
    if dry_run:
        print("MODE     : dry run -- nothing will be written")
    print()

    if settings.metadata_backend == "sqlite" and settings.vector_backend == "localfile":
        changed = _sqlite_localfile(tenant_id, visibility, doc_ids, dry_run)
    elif settings.metadata_backend == "postgres" and settings.vector_backend == "pgvector":
        changed = _postgres_pgvector(tenant_id, visibility, doc_ids, dry_run)
    else:
        raise SystemExit(
            f"unsupported backend combination {backend}. Use "
            f"POST /documents/{{id}}/reprocess instead, which rebuilds the "
            f"vector payloads through the normal pipeline."
        )

    print()
    print(f"{'would change' if dry_run else 'changed'}: {changed} document(s)")
    if changed and not dry_run:
        print("Cached answers may now be stale for this tenant; they expire via "
              "CACHE_TTL_SECONDS, or restart-safe: ingest/delete anything to bump "
              "the generation.")


if __name__ == "__main__":
    main()
