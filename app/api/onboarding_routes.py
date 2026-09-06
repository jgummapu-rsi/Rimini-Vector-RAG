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

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.auth import get_container, require_admin
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
    trusting it with no proxy in front is itself spoofable. Tracked alongside
    the rest of the ingress-hardening work (CLAUDE.md roadmap item 5)."""
    return request.client.host if request.client else "unknown"

_PBKDF2_ALGO = "sha256"
_PBKDF2_ITERATIONS = 390_000   # OWASP 2023 minimum guidance for PBKDF2-HMAC-SHA256
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


class GatewayConfigRequest(BaseModel):
    base_url: str = Field(..., max_length=500)
    api_key: str = Field(..., max_length=500)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    req: RegisterRequest, request: Request, container: Container = Depends(get_container),
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
    token = "sk-" + secrets.token_urlsafe(24)          # same scheme as scripts/seed.py
    try:
        user_id = container.metadata.create_user_with_password(
            tenant_id, email, Role.ADMIN.value, token, _hash_password(req.password))
    except EmailAlreadyRegistered:
        # The get_user_by_email check above passed, but another request for
        # the SAME email committed in between (see idx_users_email_password_unique
        # in schema.sql) -- same response as the ordinary pre-check above, just
        # reached via the race instead of the common path.
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "an account with this email already exists")

    container.metadata.write_audit(tenant_id, user_id, "onboarding_register", tenant_id,
                                    {"email": email})
    log.info("onboarding register", extra={"event": "onboarding_register", "tenant_id": tenant_id})
    return {"api_token": token}


@router.post("/login")
def login(
    req: LoginRequest, request: Request, container: Container = Depends(get_container),
) -> dict:
    """API tokens are stored as a one-way hash (never recoverable), so a
    successful login mints and returns a FRESH token rather than echoing back
    the original -- that fresh token invalidates whatever the account's
    previous token was."""
    email = req.email.strip().lower()
    ip = _client_ip(request)
    # Checked BEFORE touching the password hash: a rejected-here caller never
    # reaches the (deliberately slow) PBKDF2 verify, so the limiter also
    # bounds CPU spent on hashing, not just the eventual 401. Both `.hit()`
    # calls are made unconditionally (not `A and B` / `A or B`) so every
    # attempt is always recorded against both counters, regardless of which
    # one (if either) is already over its limit.
    email_ok = container.login_email_limiter.hit(email)
    ip_ok = container.login_ip_limiter.hit(ip)
    if not (email_ok and ip_ok):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many login attempts, try again later",
        )
    user = container.metadata.get_user_by_email(email)
    if user is None or not user.password_hash or not _verify_password(req.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    # A genuine success shouldn't leave the account looking "attacked" for the
    # rest of the window -- only the per-email counter is cleared; the per-IP
    # counter still tracks that source's overall volume.
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
    db_base_url = container.metadata.get_system_config("litellm_base_url") or ""
    db_api_key = container.metadata.get_system_config("litellm_api_key") or ""
    configured = bool(cfg.litellm_base_url or cfg.litellm_api_key or db_base_url or db_api_key)
    return {"litellm_configured": configured}


@router.post("/gateway-config")
def gateway_config(
    req: GatewayConfigRequest,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    """ADMIN-only (not public): system_config is a single global table, not
    tenant-scoped, so an unauthenticated write here would let any anonymous
    caller repoint every tenant's LLM traffic (and exfiltrate answers) to an
    attacker-controlled base_url. Costs the onboarding UI nothing extra: a
    freshly self-registered user's api_token is ADMIN-role by construction
    (see `register` above), so register -> use that token as the bearer header
    for this call works end to end with no extra step."""
    container.metadata.set_system_config("litellm_base_url", req.base_url.strip())
    container.metadata.set_system_config("litellm_api_key", req.api_key.strip())
    container.metadata.write_audit(principal.tenant_id, principal.user_id,
                                    "gateway_config_updated", "system_config", {})
    return {"ok": True}
