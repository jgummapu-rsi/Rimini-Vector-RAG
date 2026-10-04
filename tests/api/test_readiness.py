from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.adapters.postgres.db import transaction


def test_readiness_verifies_dependencies_without_recreating_schema(container):
    with TestClient(create_app(container)) as client:
        assert client.get("/readyz").json() == {"status": "ready"}
        container.cache._r.ft(container.settings.cache_index_name).dropindex()
        assert client.get("/readyz").status_code == 503
        assert (
            container.settings.cache_index_name.encode()
            not in container.cache._r.execute_command("FT._LIST")
        )
        assert client.get("/healthz").status_code == 200


def test_readiness_rejects_changed_profile_without_exposing_details(container):
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE embedding_profile SET profile_id='different-profile'")
    with TestClient(create_app(container)) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}
