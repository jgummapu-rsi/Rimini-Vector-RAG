"""Change the visibility of already-ingested documents.

Run:  python -m scripts.set_visibility <tenant_id> <private|tenant> [document_id ...]
      python -m scripts.set_visibility <tenant_id> tenant --all
      python -m scripts.set_visibility <tenant_id> tenant --all --dry-run

Why this exists
---------------
`visibility` is authoritative on `documents` and also copied into vector payloads.
This utility updates both representations and advances durable authorization
epochs without reprocessing document content. Retrieval checks current document
permissions rather than relying on an old vector payload alone.

This is an administrative command, not part of the request path.
"""

from __future__ import annotations

import logging
import sys

from app.shared.adapters.postgres.db import transaction
from app.shared.config import settings
from app.shared.domain.models import Visibility

_ALLOWED = (Visibility.PRIVATE.value, Visibility.TENANT.value)
log = logging.getLogger(__name__)


def _postgres_pgvector(tenant_id: str, visibility: str, doc_ids: list[str], dry_run: bool) -> int:

    with transaction(settings.postgres_dsn) as cur:
        if doc_ids:
            cur.execute(
                "SELECT id, filename, visibility FROM documents "
                "WHERE tenant_id=%s AND id = ANY(%s)",
                (tenant_id, doc_ids),
            )
        else:
            cur.execute(
                "SELECT id, filename, visibility FROM documents WHERE tenant_id=%s", (tenant_id,)
            )
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
            cur.execute(
                "UPDATE documents SET visibility=%s WHERE tenant_id=%s AND id=%s",
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
        raise SystemExit("pass document ids, or --all to change every document in the tenant")

    print(f"tenant   : {tenant_id}")
    print(f"target   : visibility={visibility}")
    print(f"documents: {'ALL in tenant' if take_all else ', '.join(doc_ids)}")
    if dry_run:
        print("MODE     : dry run -- nothing will be written")
    print()

    changed = _postgres_pgvector(tenant_id, visibility, doc_ids, dry_run)

    print()
    print(f"{'would change' if dry_run else 'changed'}: {changed} document(s)")
    if changed and not dry_run:
        print("Authorization updated; retrieval uses current document access and cache epochs.")


if __name__ == "__main__":
    main()
