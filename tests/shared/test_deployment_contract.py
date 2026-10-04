from pathlib import Path

import yaml


def test_runtime_containers_have_nonroot_resource_and_filesystem_bounds():
    dockerfile = Path("Dockerfile").read_text()
    assert "USER 10001:10001" in dockerfile
    assert "FROM python:3.12-slim@sha256:" in dockerfile
    assert "/readyz" in dockerfile
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    for name in ("api", "worker", "api-production", "worker-production"):
        service = compose["services"][name]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert service["pids_limit"] == 256
        assert service["mem_limit"] == "4g"
        assert service["tmpfs"]
    assert compose["services"]["api"]["ports"] == ["127.0.0.1:8000:8000"]
    for name in ("postgres", "redis"):
        assert "@sha256:" in compose["services"][name]["image"]


def test_production_profile_uses_mounted_secrets_and_preprovisioned_storage():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = compose["services"]
    for name in ("api-production", "worker-production"):
        service = services[name]
        assert service["profiles"] == ["production"]
        assert "build" not in service
        assert "depends_on" not in service
        env = service["environment"]
        assert env["INITIALIZE_SCHEMA"] == "false"
        for key, secret in (
            ("DATABASE_URL", "runtime_database"),
            ("REDIS_URL", "runtime_redis"),
            ("LITELLM_API_KEY", "gateway_key"),
        ):
            assert env[key] == ""
            assert env[key + "_FILE"] == "/run/secrets/" + secret
            assert secret in service["secrets"]
    api = services["api-production"]
    assert "--ssl-certfile" in api["command"]
    assert "--ssl-keyfile" in api["command"]
    assert api["healthcheck"]["test"] == ["CMD", "python", "-m", "scripts.tls_healthcheck"]
    for name in ("api", "worker", "postgres", "redis"):
        assert services[name]["profiles"] == ["development"]
