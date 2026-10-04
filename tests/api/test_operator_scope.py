from fastapi.testclient import TestClient

from app.api.app import create_app


def test_self_registered_admin_cannot_read_metrics_or_change_gateway(container):
    with TestClient(create_app(container)) as client:
        response = client.post(
            "/onboarding/register", json={"email": "new@example.test", "password": "valid-password"}
        )
        headers = {"Authorization": "Bearer " + response.json()["api_token"]}
        assert client.get("/metrics", headers=headers).status_code == 403
        assert client.get("/onboarding/capabilities", headers=headers).json() == {
            "configure_gateway": False
        }
        assert (
            client.post(
                "/onboarding/gateway-config",
                headers=headers,
                json={"base_url": "https://attacker.test", "api_key": "test"},
            ).status_code
            == 403
        )
        assert container.metadata.get_gateway_config() is None


def test_operator_gateway_update_is_complete_and_allowlisted(container, tenant):
    container.settings.operator_user_ids = [tenant["admin_id"]]
    container.settings.gateway_allowed_origins = ["https://gateway.test"]
    headers = {"Authorization": "Bearer " + tenant["admin_token"]}
    with TestClient(create_app(container)) as client:
        assert client.get("/onboarding/capabilities", headers=headers).json() == {
            "configure_gateway": True
        }
        for url, key in (
            ("https://evil.test", "key"),
            ("http://gateway.test", "key"),
            ("https://gateway.test", ""),
            ("https://gateway.test/path", "key"),
        ):
            assert (
                client.post(
                    "/onboarding/gateway-config",
                    headers=headers,
                    json={"base_url": url, "api_key": key},
                ).status_code
                == 400
            )
            assert container.metadata.get_gateway_config() is None
        assert (
            client.post(
                "/onboarding/gateway-config",
                headers=headers,
                json={"base_url": "https://gateway.test", "api_key": "key"},
            ).status_code
            == 200
        )
        assert container.gateway._resolve_config() == ("https://gateway.test", "key")
