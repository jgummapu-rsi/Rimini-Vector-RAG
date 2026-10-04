"""Helpers shared by ingest_routes.py and retrieval_routes.py."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile, status

from app.retrieval.rag.access import can_view
from app.shared.domain.models import Document, Principal, Role

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
    payload = {
        "user_id": doc.owner_user_id,
        "visibility": doc.visibility,
        "acl_user_ids": doc.acl_user_ids,
        "scope": doc.scope,
    }
    if not can_view(payload, principal.user_id, principal.role.value):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


def _owned_or_404(doc: Document, principal: Principal) -> None:
    """Read access to a scope=global document does not imply write/delete access --
    only the owning (platform) tenant may reprocess/delete its own documents."""
    if doc.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


def _resolve_target_document(
    container, principal: Principal, document_id: str | None, filename: str
) -> Document | None:
    """Which already-ingested document, if any, is this upload a new version of?

    Two ways to answer, in precedence order:

    1. An explicit `document_id` on the request -- the caller is telling us
       outright. Authorized with `_visible_or_404` (you may not act on a
       document you can't see) and `_owned_or_404` (reading a global document
       doesn't let you replace it), plus a third check `/reprocess` does NOT
       make: the caller must be the document's owner, or a tenant admin.

       That extra check is the difference between the two operations. Reprocess
       re-runs the pipeline over bytes that do not change, so letting any
       member trigger it on a tenant-visible document is harmless. Replacing
       the CONTENT of a document shared with the whole tenant is not -- it lets
       one member rewrite what everyone else retrieves, under the original
       author's name. Ownership is the right boundary for a write.

       A bad id is a 404, never a silent fall-through to "create a new
       document", because the caller asked for something specific.

    2. Otherwise, the caller's own newest document with the same filename. This
       is what makes re-dragging `handbook.pdf` onto the trace UI do the
       obvious thing with no client changes. Deliberately owner-scoped inside
       the store (see `MetadataStore.get_document_by_filename`): a shared
       filename is unremarkable, so a tenant-wide match would let one member
       overwrite a colleague's private document.

    Returns None when nothing matches -- that is the ordinary "this is a new
    document" case, not an error.
    """
    if document_id:
        doc = container.metadata.get_document(principal.tenant_id, document_id)
        if doc is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        _visible_or_404(doc, principal)
        _owned_or_404(doc, principal)
        if doc.owner_user_id != principal.user_id and principal.role != Role.ADMIN:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        return doc
    return container.metadata.get_document_by_filename(
        principal.tenant_id, principal.user_id, filename
    )
