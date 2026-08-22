"""AuthN/AuthZ: resolve the bearer token to a Principal (tenant + user + role).

Uses FastAPI's HTTPBearer security scheme so Swagger UI shows an "Authorize"
button: paste just the token (no "Bearer " prefix) and it's applied to every
request. The raw header is still accepted for curl/clients that send it directly.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.container import Container
from app.domain.models import Principal
from app.observability import bind

# auto_error=False so we return our own 401 (and don't hard-require it at parse time)
_bearer = HTTPBearer(auto_error=False, description="Paste your API token (api_token).")


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    container: Container = Depends(get_container),
) -> Principal:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = credentials.credentials.strip()
    principal = container.metadata.get_principal_by_token(token)
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    # attach tenant/user to the request's log context (reset by the middleware)
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id,
         role=principal.role.value)
    return principal


def require_ingest(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.can_ingest():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "role cannot ingest")
    return principal


def require_delete(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.can_delete():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "role cannot delete")
    return principal
