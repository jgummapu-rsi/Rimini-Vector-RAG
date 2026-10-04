"""Self-service onboarding: email/password signup+login (an alternative to
scripts/seed.py for getting a tenant+admin+api_token), plus a live-editable
LiteLLM gateway config gate.

Password hashing: PBKDF2-HMAC-SHA256 via hashlib (stdlib, no new dependency --
requirements.txt has neither bcrypt nor passlib). Stored format:
"pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>", self-describing so the
iteration count can be raised later without invalidating existing hashes.

Enumeration note: /login returns the same 401 for "unknown email" and "wrong
password", to make it slightly harder for a caller to enumerate registered
emails (register still reports "already exists" distinctly -- a real user
needs that signal to know to switch to login).

Rate limiting: /login is bounded per-email AND per-IP, /register per-IP (see
app.shared.rate_limit.RateLimiter, wired up in app.shared.container). Per-email
bounds credential-stuffing against one account regardless of source IP;
per-IP additionally bounds one source hammering many different emails or
mass-creating accounts. It is in-process/per-API-instance (see that module's
docstring for the horizontal-scaling caveat) -- still real protection for
today's single-process deployment, not a placeholder.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.auth import get_container, get_principal, require_admin, require_operator
from app.shared.container import Container
from app.shared.domain.models import Principal, Role
from app.shared.ports.metadata_store import EmailAlreadyRegistered

router = APIRouter(prefix="/onboarding", tags=["onboarding"])
log = logging.getLogger("onboarding")


def _client_ip(request: Request) -> str:
    """Best-effort caller address for rate-limiting. This deployment has no
    reverse proxy in front of it yet (see docker-compose.yml/README): a real
    one terminating TLS in front of the API would need to set and be trusted
    for X-Forwarded-For, and this would need to read that header instead --
    trusting it with no proxy in front is itself spoofable."""
    return request.client.host if request.client else "unknown"


_PBKDF2_ALGO = "sha256"
_PBKDF2_ITERATIONS = 390_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations_s, salt_hex, hash_hex = stored.split("$")
        iterations, salt = int(iterations_s), bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate.hex(), hash_hex)


class RegisterRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., min_length=8, max_length=200)
    display_name: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=200)


class CreateMemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(..., max_length=320)
    password: str = Field(..., min_length=8, max_length=200)
    role: Role = Role.MEMBER


@router.get("/me")
def workspace_identity(
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    return {
        **container.metadata.get_workspace_identity(principal),
        "role": principal.role.value,
        "manage_team": principal.role == Role.ADMIN,
        "can_ingest": principal.can_ingest(),
        "publish_global": (
            principal.role == Role.ADMIN
            and principal.tenant_id == container.settings.platform_tenant_id
        ),
        "configure_gateway": (
            principal.role == Role.ADMIN
            and principal.user_id in container.settings.operator_user_ids
        ),
    }


@router.get("/members")
def list_members(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    return container.metadata.list_tenant_members(principal.tenant_id, limit, offset)


@router.delete("/members/{member_id}")
def delete_member(
    member_id: str,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    try:
        email = container.metadata.delete_tenant_member(principal, member_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"email": email, "status": "deleted"}


@router.post("/members", status_code=status.HTTP_201_CREATED)
def create_member(
    req: CreateMemberRequest,
    request: Request,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    """Provision a login in the caller's tenant; tenant identity is never client supplied."""
    if not container.register_ip_limiter.hit(_client_ip(request)):
        raise HTTPException(
            429,
            "Too many accounts were created. Wait a few minutes and try again.",
        )
    email = req.email.strip().lower()
    if not _EMAIL_RE.fullmatch(email):
        raise HTTPException(400, "Enter a valid email address, such as dana@meridianhealth.com.")
    if container.metadata.get_user_by_email(email) is not None:
        raise HTTPException(409, "This email is already registered. Use a different email address.")
    try:
        member_id = container.metadata.create_user_with_password(
            principal.tenant_id,
            email,
            req.role.value,
            "sk-" + secrets.token_urlsafe(24),
            _hash_password(req.password),
        )
    except EmailAlreadyRegistered as exc:
        raise HTTPException(
            409,
            "This email is already registered. Use a different email address.",
        ) from exc
    container.metadata.write_audit(
        principal.tenant_id,
        principal.user_id,
        "team_member_created",
        member_id,
        {"email": email, "role": req.role.value},
    )
    return {"email": email, "role": req.role.value, "status": "active"}


class GatewayConfigRequest(BaseModel):
    base_url: str = Field(..., max_length=500)
    api_key: str = Field(..., max_length=500)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    req: RegisterRequest,
    request: Request,
    container: Container = Depends(get_container),
) -> dict:
    if not container.register_ip_limiter.hit(_client_ip(request)):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many signups from this address, try again later",
        )
    email = req.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid email")
    if container.metadata.get_user_by_email(email) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists")

    tenant_name = (req.display_name or email.split("@")[0]).strip()[:200] or email
    tenant_id = container.metadata.create_tenant(tenant_name)
    token = "sk-" + secrets.token_urlsafe(24)
    try:
        user_id = container.metadata.create_user_with_password(
            tenant_id, email, Role.ADMIN.value, token, _hash_password(req.password)
        )
    except EmailAlreadyRegistered:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "an account with this email already exists"
        ) from None

    container.metadata.write_audit(
        tenant_id, user_id, "onboarding_register", tenant_id, {"email": email}
    )
    log.info("onboarding register", extra={"event": "onboarding_register", "tenant_id": tenant_id})
    return {"api_token": token}


@router.post("/login")
def login(
    req: LoginRequest,
    request: Request,
    container: Container = Depends(get_container),
) -> dict:
    """API tokens are stored as a one-way hash (never recoverable), so a
    successful login mints and returns a FRESH token rather than echoing back
    the original -- that fresh token invalidates whatever the account's
    previous token was."""
    email = req.email.strip().lower()
    ip = _client_ip(request)

    email_ok = container.login_email_limiter.hit(email)
    ip_ok = container.login_ip_limiter.hit(ip)
    if not (email_ok and ip_ok):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many login attempts, try again later",
        )
    user = container.metadata.get_user_by_email(email)
    if (
        user is None
        or not user.password_hash
        or not _verify_password(req.password, user.password_hash)
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")

    container.login_email_limiter.reset(email)
    token = "sk-" + secrets.token_urlsafe(24)
    container.metadata.rotate_api_token(user.id, token)
    container.metadata.write_audit(user.tenant_id, user.id, "onboarding_login", user.id, {})
    return {"api_token": token}


@router.get("/status")
def onboarding_status(container: Container = Depends(get_container)) -> dict:
    """Public: whether a chat/vision LLM call would succeed right now, from
    EITHER an env-set LITELLM_BASE_URL/KEY or a DB override. Drives the
    onboarding UI's "configure your gateway" gate."""
    cfg = container.settings
    pair = container.metadata.get_gateway_config()
    configured = (
        bool(all(pair)) if pair is not None else bool(cfg.litellm_base_url and cfg.litellm_api_key)
    )
    return {"litellm_configured": configured}


@router.post("/gateway-config")
def gateway_config(
    req: GatewayConfigRequest,
    principal: Principal = Depends(require_operator),
    container: Container = Depends(get_container),
) -> dict:
    """ADMIN-only (not public): system_config is a single global table, not
    tenant-scoped, so an unauthenticated write here would let any anonymous
    caller repoint every tenant's LLM traffic (and exfiltrate answers) to an
    attacker-controlled base_url. Costs the onboarding UI nothing extra: a
    freshly self-registered user's api_token is ADMIN-role by construction
    (see `register` above), so register -> use that token as the bearer header
    for this call works end to end with no extra step."""
    base_url = req.base_url.strip().rstrip("/")
    api_key = req.api_key.strip()
    parsed = urlsplit(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path
        or origin not in container.settings.gateway_allowed_origins
        or not api_key
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "gateway requires an allowed HTTPS origin and a nonempty key",
        )
    container.metadata.set_gateway_config(base_url, api_key)
    container.metadata.write_audit(
        principal.tenant_id, principal.user_id, "gateway_config_updated", "system_config", {}
    )
    return {"ok": True}


@router.get("/capabilities")
def capabilities(
    principal: Principal = Depends(get_principal), container: Container = Depends(get_container)
) -> dict:
    return {
        "configure_gateway": principal.role == Role.ADMIN
        and principal.user_id in container.settings.operator_user_ids
    }
