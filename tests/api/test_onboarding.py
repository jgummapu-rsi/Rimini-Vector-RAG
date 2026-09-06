"""Self-service onboarding: POST /onboarding/register, /login, GET /status,
POST /gateway-config. See app.api.onboarding_routes.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_register_returns_a_usable_admin_token(client):
    r = client.post("/onboarding/register",
                    json={"email": "dev@example.test", "password": "hunter2hunter"})
    assert r.status_code == 201
    token = r.json()["api_token"]
    assert token.startswith("sk-")

    # the token immediately authenticates, as an admin (can read /documents,
    # which requires get_principal, and can call the admin-only endpoint below)
    docs = client.get("/documents", headers=_auth(token))
    assert docs.status_code == 200


def test_register_duplicate_email_is_rejected(client):
    client.post("/onboarding/register",
               json={"email": "dupe@example.test", "password": "hunter2hunter"})
    r = client.post("/onboarding/register",
                    json={"email": "dupe@example.test", "password": "different1"})
    assert r.status_code == 409


def test_login_with_correct_password_returns_a_fresh_working_token(client):
    """Tokens are stored as a one-way hash, so login can't echo the original
    back -- it mints a new one instead, which must also invalidate the old one."""
    reg = client.post("/onboarding/register",
                      json={"email": "returning@example.test", "password": "correcthorse"})
    old_token = reg.json()["api_token"]

    login = client.post("/onboarding/login",
                        json={"email": "returning@example.test", "password": "correcthorse"})
    assert login.status_code == 200
    new_token = login.json()["api_token"]
    assert new_token != old_token

    assert client.get("/documents", headers={"Authorization": f"Bearer {new_token}"}).status_code == 200
    old_check = client.get("/documents", headers={"Authorization": f"Bearer {old_token}"})
    assert old_check.status_code == 401


def test_login_with_wrong_password_is_rejected(client):
    client.post("/onboarding/register",
               json={"email": "wrongpw@example.test", "password": "correcthorse"})
    r = client.post("/onboarding/login",
                    json={"email": "wrongpw@example.test", "password": "wrong-password"})
    assert r.status_code == 401


def test_login_with_unknown_email_gets_same_status_as_wrong_password(client):
    r = client.post("/onboarding/login",
                    json={"email": "nobody@example.test", "password": "whatever1"})
    assert r.status_code == 401


def test_status_false_by_default_true_after_gateway_config(client):
    reg = client.post("/onboarding/register",
                      json={"email": "gw@example.test", "password": "hunter2hunter"})
    token = reg.json()["api_token"]

    before = client.get("/onboarding/status")
    assert before.json() == {"litellm_configured": False}

    set_cfg = client.post("/onboarding/gateway-config",
                          json={"base_url": "https://gw.example.com", "api_key": "sk-gw"},
                          headers=_auth(token))
    assert set_cfg.status_code == 200

    after = client.get("/onboarding/status")
    assert after.json() == {"litellm_configured": True}


def test_gateway_config_requires_auth(client):
    r = client.post("/onboarding/gateway-config",
                    json={"base_url": "https://gw.example.com", "api_key": "sk-gw"})
    assert r.status_code == 401


def test_gateway_config_requires_admin_role(client, tenant):
    r = client.post("/onboarding/gateway-config",
                    json={"base_url": "https://gw.example.com", "api_key": "sk-gw"},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 403


# ------------------------------------------------------------ rate limiting --

def test_login_is_rate_limited_per_email(client, container):
    container.login_email_limiter._max_hits = 3
    client.post("/onboarding/register",
                json={"email": "ratelimited@example.test", "password": "hunter2hunter"})

    for _ in range(3):
        r = client.post("/onboarding/login",
                        json={"email": "ratelimited@example.test", "password": "wrong-password"})
        assert r.status_code == 401

    r = client.post("/onboarding/login",
                    json={"email": "ratelimited@example.test", "password": "wrong-password"})
    assert r.status_code == 429

    # even the CORRECT password is now rejected -- the limit is on attempt
    # volume, not on wrong guesses, so it can't be bypassed by a lucky guess
    r = client.post("/onboarding/login",
                    json={"email": "ratelimited@example.test", "password": "hunter2hunter"})
    assert r.status_code == 429


def test_login_rate_limit_is_scoped_per_email_not_global(client, container):
    container.login_email_limiter._max_hits = 1
    client.post("/onboarding/register",
                json={"email": "alice-rl@example.test", "password": "hunter2hunter"})
    client.post("/onboarding/register",
                json={"email": "bob-rl@example.test", "password": "hunter2hunter"})

    r1 = client.post("/onboarding/login",
                     json={"email": "alice-rl@example.test", "password": "wrong-password"})
    assert r1.status_code == 401
    r2 = client.post("/onboarding/login",
                     json={"email": "alice-rl@example.test", "password": "wrong-password"})
    assert r2.status_code == 429

    # bob is unaffected by alice's limit being tripped
    r3 = client.post("/onboarding/login",
                     json={"email": "bob-rl@example.test", "password": "hunter2hunter"})
    assert r3.status_code == 200


def test_successful_login_resets_the_email_rate_limit(client, container):
    container.login_email_limiter._max_hits = 2
    client.post("/onboarding/register",
                json={"email": "recover-rl@example.test", "password": "hunter2hunter"})

    client.post("/onboarding/login",
               json={"email": "recover-rl@example.test", "password": "wrong-password"})
    ok = client.post("/onboarding/login",
                     json={"email": "recover-rl@example.test", "password": "hunter2hunter"})
    assert ok.status_code == 200

    # the counter was reset by the success above, so this genuinely-wrong
    # attempt is a fresh first strike, not an already-exhausted budget
    r = client.post("/onboarding/login",
                    json={"email": "recover-rl@example.test", "password": "wrong-password"})
    assert r.status_code == 401


def test_login_is_also_rate_limited_per_ip(client, container):
    container.login_ip_limiter._max_hits = 2
    for email in ("ip-a@example.test", "ip-b@example.test", "ip-c@example.test"):
        client.post("/onboarding/register",
                    json={"email": email, "password": "hunter2hunter"})

    for email in ("ip-a@example.test", "ip-b@example.test"):
        r = client.post("/onboarding/login",
                        json={"email": email, "password": "wrong-password"})
        assert r.status_code == 401

    # a THIRD distinct email from the same (test client's) IP still trips it
    r = client.post("/onboarding/login",
                    json={"email": "ip-c@example.test", "password": "wrong-password"})
    assert r.status_code == 429


def test_register_is_rate_limited_per_ip(client, container):
    container.register_ip_limiter._max_hits = 2
    for i in range(2):
        r = client.post("/onboarding/register",
                        json={"email": f"reg-rl-{i}@example.test", "password": "hunter2hunter"})
        assert r.status_code == 201

    r = client.post("/onboarding/register",
                    json={"email": "reg-rl-3@example.test", "password": "hunter2hunter"})
    assert r.status_code == 429
