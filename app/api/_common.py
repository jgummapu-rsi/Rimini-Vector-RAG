"""Helpers shared by ingest_routes.py and retrieval_routes.py."""
from __future__ import annotations

from fastapi import HTTPException, UploadFile, status

from app.shared.domain.models import Document, Principal
from app.retrieval.rag.access import can_view

# Read granularity for uploads. Small enough that the cap below is enforced
# before a runaway body can matter, large enough not to make normal reads chatty.
_UPLOAD_CHUNK = 1024 * 1024


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing to accumulate more than `max_bytes` in memory.

    The obvious `await file.read()` reads the WHOLE body first and only then
    compares its length to the limit -- which means the limit is enforced only
    after the damage (a multi-GB allocation) is already done. Reading in bounded
    chunks and stopping at the cap makes the limit actually bounding. The
    Content-Length guard in app.api.app rejects most oversized requests earlier;
    this covers the case where that header is absent or understated.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"file exceeds {max_bytes // (1024 * 1024)} MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _visible_or_404(doc: Document, principal: Principal) -> None:
    """Raise 404 (not 403 -- existence is itself sensitive) unless `principal` can view `doc`."""
    payload = {"user_id": doc.owner_user_id, "visibility": doc.visibility,
               "acl_user_ids": doc.acl_user_ids, "scope": doc.scope}
    if not can_view(payload, principal.user_id, principal.role.value):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


def _owned_or_404(doc: Document, principal: Principal) -> None:
    """Read access to a scope=global document does not imply write/delete access --
    only the owning (platform) tenant may reprocess/delete its own documents."""
    if doc.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
