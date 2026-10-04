"""Tenant directory and account provisioning use real storage and auth boundaries."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Role


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as client:
        yield client


def auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("role", ["admin", "member", "viewer"])
def test_create_member_can_sign_in_to_same_tenant(client, container, tenant, other_tenant, role):
    payload = {"email": "  NEW@Example.test ", "password": "correct-horse-42", "role": role}
    response = client.post("/onboarding/members", json=payload, headers=auth(tenant["admin_token"]))
    assert response.status_code == 201
    assert response.json() == {"email": "new@example.test", "role": role, "status": "active"}
    login = client.post(
        "/onboarding/login",
        json={"email": "new@example.test", "password": payload["password"]},
    )
    assert login.status_code == 200
    principal = container.metadata.get_principal_by_token(login.json()["api_token"])
    assert principal.tenant_id == tenant["id"]
    assert principal.role == Role(role)
    directory = client.get("/onboarding/members", headers=auth(other_tenant["admin_token"])).json()
    assert "new@example.test" not in str(directory)
    with transaction(container.metadata.dsn) as cur:
        cur.execute("SELECT password_hash, api_token FROM users WHERE id=%s", (principal.user_id,))
        stored = cur.fetchone()
        assert stored["password_hash"] != payload["password"]
        assert stored["api_token"] != login.json()["api_token"]


@pytest.mark.parametrize("token_key", ["member_token", "viewer_token", None])
def test_only_admins_can_access_team(client, tenant, token_key):
    headers = auth(tenant[token_key]) if token_key else {}
    expected = 403 if token_key else 401
    assert client.get("/onboarding/members", headers=headers).status_code == expected
    response = client.post(
        "/onboarding/members",
        headers=headers,
        json={"email": "denied@example.test", "password": "correct-horse"},
    )
    assert response.status_code == expected


def test_directory_is_paginated_scoped_and_has_no_credentials(client, tenant, other_tenant):
    headers = auth(tenant["admin_token"])
    first = client.get("/onboarding/members?limit=2", headers=headers).json()
    second = client.get("/onboarding/members?limit=2&offset=2", headers=headers).json()
    assert first["total"] == second["total"] == 3
    members = first["members"] + second["members"]
    assert len({member["id"] for member in members}) == 3
    for member in members:
        assert set(member) == {"id", "email", "role", "status", "created_at"}
        assert member["email"].endswith("@acme.test")
    assert client.get("/onboarding/members?limit=101", headers=headers).status_code == 422
    assert client.get("/onboarding/members?offset=-1", headers=headers).status_code == 422


def test_cannot_choose_another_tenant_or_invalid_role(client, tenant, other_tenant):
    headers = auth(tenant["admin_token"])
    base = {"email": "new@example.test", "password": "correct-horse"}
    for extra in ({"tenant_id": other_tenant["id"]}, {"role": "operator"}, {"password": "short"}):
        response = client.post("/onboarding/members", headers=headers, json={**base, **extra})
        assert response.status_code == 422
    response = client.post("/onboarding/members", headers=headers, json={**base, "email": "bad"})
    assert response.status_code == 400


def test_duplicate_email_does_not_move_or_modify_account(client, container, tenant, other_tenant):
    payload = {"email": "same@example.test", "password": "correct-horse", "role": "viewer"}
    created = client.post(
        "/onboarding/members",
        headers=auth(other_tenant["admin_token"]),
        json=payload,
    )
    assert created.status_code == 201
    duplicate = client.post(
        "/onboarding/members",
        headers=auth(tenant["admin_token"]),
        json={**payload, "role": "admin"},
    )
    assert duplicate.status_code == 409
    member = container.metadata.get_user_by_email(payload["email"])
    assert member.tenant_id == other_tenant["id"]
    assert member.role == "viewer"


def test_workspace_identity_respects_role(client, tenant):
    for role in ("admin", "member", "viewer"):
        response = client.get("/onboarding/me", headers=auth(tenant[f"{role}_token"]))
        assert response.status_code == 200
        assert response.json() == {
            "email": f"{role}@acme.test",
            "organization": "Acme",
            "role": role,
            "manage_team": role == "admin",
            "can_ingest": role != "viewer",
            "configure_gateway": False,
            "publish_global": False,
        }


def test_member_creation_is_rate_limited(client, container, tenant):
    container.register_ip_limiter._max_hits = 1
    headers = auth(tenant["admin_token"])
    first = client.post(
        "/onboarding/members",
        headers=headers,
        json={"email": "one@example.test", "password": "correct-horse"},
    )
    assert first.status_code == 201
    limited = client.post(
        "/onboarding/members",
        headers=headers,
        json={"email": "two@example.test", "password": "correct-horse"},
    )
    assert limited.status_code == 429


def test_delete_revokes_login_and_token_but_preserves_document_owner(client, container, tenant):
    headers = auth(tenant["admin_token"])
    credentials = {"email": "remove@example.test", "password": "correct-horse"}
    client.post("/onboarding/members", headers=headers, json=credentials)
    token = client.post("/onboarding/login", json=credentials).json()["api_token"]
    member = container.metadata.get_principal_by_token(token)
    upload = client.post(
        "/ingest",
        headers=auth(token),
        files={"file": ("retained.txt", b"A retained organization document.", "text/plain")},
        data={"visibility": "tenant"},
    )
    assert upload.status_code in (200, 202)
    response = client.delete(f"/onboarding/members/{member.user_id}", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"email": credentials["email"], "status": "deleted"}
    assert client.get("/onboarding/me", headers=auth(token)).status_code == 401
    assert client.post("/onboarding/login", json=credentials).status_code == 401
    directory = client.get("/onboarding/members", headers=headers).json()
    assert all(row["id"] != member.user_id for row in directory["members"])
    document = container.metadata.get_document(tenant["id"], upload.json()["document_id"])
    assert document.owner_user_id == member.user_id
    with transaction(container.metadata.dsn) as cur:
        cur.execute("SELECT status, password_hash FROM users WHERE id=%s", (member.user_id,))
        assert dict(cur.fetchone()) == {"status": "deleted", "password_hash": None}
        cur.execute(
            "SELECT target FROM audit_log WHERE action='team_member_deleted' AND tenant_id=%s",
            (tenant["id"],),
        )
        assert cur.fetchone()["target"] == member.user_id
    # Re-adding an email creates a new identity, not access to the deleted account.
    assert client.post("/onboarding/members", headers=headers, json=credentials).status_code == 201
    assert container.metadata.get_user_by_email(credentials["email"]).id != member.user_id


def test_delete_is_scoped_and_cannot_remove_self(client, tenant, other_tenant):
    target = f"/onboarding/members/{tenant['member_id']}"
    assert client.delete(target).status_code == 401
    for role in ("member", "viewer"):
        assert client.delete(target, headers=auth(tenant[f"{role}_token"])).status_code == 403
    assert client.delete(target, headers=auth(other_tenant["admin_token"])).status_code == 404
    headers = auth(tenant["admin_token"])
    assert (
        client.delete(
            f"/onboarding/members/{tenant['admin_id']}",
            headers=headers,
        ).status_code
        == 409
    )
    assert client.delete(target, headers=headers).status_code == 200
    assert client.delete(target, headers=headers).status_code == 404
    assert client.get("/onboarding/me", headers=headers).status_code == 200


def test_concurrent_admin_deletions_keep_one_active_admin(container, tenant):

    second_id = container.metadata.create_user(
        tenant["id"],
        "second@example.test",
        "admin",
        "second",
    )
    first = container.metadata.get_principal_by_token(tenant["admin_token"])
    second = container.metadata.get_principal_by_token("second")

    def remove(actor, target):
        try:
            container.metadata.delete_tenant_member(actor, target)
            return "deleted"
        except PermissionError:
            return "revoked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            executor.submit(remove, first, second_id),
            executor.submit(remove, second, first.user_id),
        ]
        assert sorted(result.result() for result in results) == ["deleted", "revoked"]
    with transaction(container.metadata.dsn) as cur:
        cur.execute(
            "SELECT count(*) AS total FROM users WHERE tenant_id=%s "
            "AND role='admin' AND status='active'",
            (tenant["id"],),
        )
        assert cur.fetchone()["total"] == 1
