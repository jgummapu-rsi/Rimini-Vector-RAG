"""Document access rules (visibility / ACL), enforced at retrieval time.

A record (document) is visible to a user when ANY of:
  - it is scope == "global"         (bypasses everything below; readable by any
                                      authenticated user, in any tenant)
  - the user's role is admin        (admins see all docs in their tenant)
  - the user owns it                (user_id == record.user_id)
  - it is tenant-wide               (visibility == "tenant")
  - it is shared to the user        (visibility == "shared" AND user in acl_user_ids)

Tenant scoping is enforced separately (the store only ever loads one tenant's
records, plus any scope=global records from other tenants); this adds the
intra-tenant per-user filter.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.shared.domain.models import Principal, Role, Scope, Visibility


def can_view(payload: dict, user_id: str, role: str) -> bool:
    """Whether a document/chunk `payload` is visible to `user_id` with `role`."""
    if payload.get("scope") == Scope.GLOBAL.value:
        return True
    if role == Role.ADMIN.value:
        return True
    if payload.get("user_id") == user_id:
        return True
    if payload.get("visibility") == Visibility.TENANT.value:
        return True
    if (payload.get("visibility") == Visibility.SHARED.value
            and user_id in (payload.get("acl_user_ids") or [])):
        return True
    return False


@dataclass(frozen=True)
class AccessFilter:
    """The `can_view` rule bound to one principal.

    It is CALLABLE, so every consumer that just wants a `predicate(payload)`
    (the `VectorStore.search` port, the localfs adapter) is unchanged. It also
    exposes `user_id`/`role` so a store that can push the rule down into its own
    query language does not have to reverse-engineer a closure.

    That pushdown matters for correctness, not just speed: a store that applies
    a Python-side predicate AFTER its own LIMIT is filtering a page it has
    already truncated, so a user surrounded by other people's private documents
    can get an empty result set while visible chunks exist just past the cut.
    Filtering before the LIMIT is what makes top-k mean "top k *visible*".
    """
    user_id: str
    role: str
    document_ids: frozenset[str] | None = None

    def __call__(self, payload: dict) -> bool:
        """Whether `payload` is visible under this bound principal."""
        if self.document_ids is not None and payload.get("_id") not in self.document_ids:
            return False
        return can_view(payload, self.user_id, self.role)

    @property
    def sees_everything(self) -> bool:
        """True when the rule admits every row in the tenant, so a store can
        skip the filter entirely rather than emit a tautology."""
        return self.role == Role.ADMIN.value and self.document_ids is None


def access_predicate(
    principal: Principal, document_ids: list[str] | None = None
) -> AccessFilter:
    """Build the bound `AccessFilter` for a given principal."""
    selected = frozenset(document_ids) if document_ids is not None else None
    return AccessFilter(
        user_id=principal.user_id,
        role=principal.role.value,
        document_ids=selected,
    )
