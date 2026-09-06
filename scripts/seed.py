"""Seed a tenant + admin user and print a bearer token to test the API.

Run:  python -m scripts.seed  [tenant_name] [email]
      python -m scripts.seed --platform [tenant_name] [email]

`--platform` seeds the tenant that owns the firm-wide global knowledge base: set
the printed tenant_id as PLATFORM_TENANT_ID in .env, then that tenant's admin can
POST /ingest with scope=global to publish documents visible to every tenant.
"""
from __future__ import annotations

import secrets
import sys

from app.shared.container import build_container
from app.shared.domain.models import Role


def main() -> None:
    args = sys.argv[1:]
    is_platform = "--platform" in args
    if is_platform:
        args.remove("--platform")
    tenant_name = args[0] if len(args) > 0 else ("Our Firm" if is_platform else "Acme")
    email = args[1] if len(args) > 1 else ("admin@platform.test" if is_platform else "admin@acme.test")

    c = build_container()
    tenant_id = c.metadata.create_tenant(tenant_name)
    token = "sk-" + secrets.token_urlsafe(24)
    user_id = c.metadata.create_user(tenant_id, email, Role.ADMIN.value, token)

    print("Seeded tenant + admin user.")
    print(f"  tenant_id : {tenant_id}")
    print(f"  user_id   : {user_id}")
    print(f"  role      : admin")
    print(f"  API token : {token}")
    print()
    if is_platform:
        print("This is the PLATFORM tenant. Set in .env:")
        print(f"  PLATFORM_TENANT_ID={tenant_id}")
        print("Then this admin can publish firm-wide docs:")
        print(f'  curl -H "Authorization: Bearer {token}" -F "file=@doc.pdf" '
              f'-F "scope=global" http://127.0.0.1:8000/ingest')
    else:
        print("Test it:")
        print(f'  curl -H "Authorization: Bearer {token}" http://127.0.0.1:8000/healthz')


if __name__ == "__main__":
    main()
