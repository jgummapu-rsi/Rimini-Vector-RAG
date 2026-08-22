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

from typing import Callable

from app.domain.models import Principal, Role, Scope, Visibility


def can_view(payload: dict, user_id: str, role: str) -> bool:
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


def access_predicate(principal: Principal) -> Callable[[dict], bool]:
    uid, role = principal.user_id, principal.role.value
    return lambda payload: can_view(payload, uid, role)
