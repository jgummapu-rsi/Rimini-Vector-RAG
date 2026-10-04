from __future__ import annotations

import json
import logging
import os
import secrets
import ssl
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import parse_dsn

from app.shared import container as composition
from scripts.disposable_storage import evaluation_settings
from scripts.grant_runtime import grant_runtime

log = logging.getLogger(__name__)


def docker(*args, timeout=120):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[:1500]}")
    return (result.stdout + (result.stderr if args[0] == "logs" else "")).strip()


def main():
    image = os.environ.get("ACCEPTANCE_IMAGE", "rag-hardening-acceptance:local")
    with (
        evaluation_settings() as cfg,
        TemporaryDirectory(prefix="deployment-acceptance-") as temporary,
    ):
        root = Path(temporary)
        root.chmod(0o755)
        identity = uuid4().hex[:12]
        api, worker = "acceptance-api-" + identity, "acceptance-worker-" + identity
        blob_volume, model_volume = "acceptance-blobs-" + identity, "acceptance-models-" + identity
        role = "acceptance_" + identity
        password = secrets.token_urlsafe(32)
        params = parse_dsn(cfg.postgres_dsn)
        schema = params["dbname"]
        with psycopg2.connect(cfg.postgres_dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN PASSWORD %s NOSUPERUSER NOCREATEDB NOCREATEROLE"
                ).format(sql.Identifier(role)),
                (password,),
            )
            cursor.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(params["dbname"]), sql.Identifier(role)
                )
            )
        database = f"postgresql://{role}:{password}@host.docker.internal:{params['port']}/{params['dbname']}?options=-csearch_path%3D{schema},public"
        (root / "database").write_text(database)
        (root / "redis").write_text(cfg.redis_url.replace("127.0.0.1", "host.docker.internal"))
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-keyout",
                str(root / "key.pem"),
                "-out",
                str(root / "cert.pem"),
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        for path in root.iterdir():
            path.chmod(0o644)
        started = time.monotonic()
        try:
            migrated = composition.build_container(
                cfg.model_copy(update={"reranker_provider": "none"})
            )
            migrated.close()
            grant_runtime(cfg.postgres_dsn, schema, role)
            for volume in (blob_volume, model_volume):
                docker("volume", "create", volume)
            common = [
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "256",
                "--memory",
                "4g",
                "--cpus",
                "2",
                "--tmpfs",
                "/tmp:rw,size=256m,mode=1777",
                "--mount",
                f"type=bind,src={root},dst=/run/secrets,readonly",
                "--mount",
                f"type=volume,src={blob_volume},dst=/app/data",
                "--mount",
                f"type=volume,src={model_volume},dst=/app/models",
                "--env",
                "DATABASE_URL_FILE=/run/secrets/database",
                "--env",
                "REDIS_URL_FILE=/run/secrets/redis",
                "--env",
                f"CACHE_INDEX_NAME={cfg.cache_index_name}",
                "--env",
                "RERANKER_PROVIDER=none",
                "--env",
                "INITIALIZE_SCHEMA=false",
            ]
            docker(
                "run",
                "--detach",
                "--name",
                api,
                *common,
                "--publish",
                "127.0.0.1::8443",
                image,
                "uvicorn",
                "app.api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8443",
                "--ssl-keyfile",
                "/run/secrets/key.pem",
                "--ssl-certfile",
                "/run/secrets/cert.pem",
            )
            port = docker("port", api, "8443/tcp").rsplit(":", 1)[1]
            context = ssl.create_default_context(cafile=str(root / "cert.pem"))
            with httpx.Client(
                base_url=f"https://localhost:{port}", verify=context, timeout=10
            ) as client:
                deadline = time.monotonic() + 180
                while True:
                    try:
                        if client.get("/readyz").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "TLS API did not become ready: " + docker("logs", api)[-1500:]
                        )
                    time.sleep(2)
                registered = client.post(
                    "/onboarding/register",
                    json={
                        "email": "acceptance@example.test",
                        "password": secrets.token_urlsafe(20),
                    },
                )
                registered.raise_for_status()
                headers = {"Authorization": "Bearer " + registered.json()["api_token"]}
                content = b"Invoice 00123 amount 007.00 USD. Approval required."
                uploaded = client.post(
                    "/ingest", headers=headers, files={"file": ("invoice.txt", content)}
                )
                uploaded.raise_for_status()
                job = uploaded.json()["job_id"]
                docker(
                    "run",
                    "--detach",
                    "--name",
                    worker,
                    *common,
                    image,
                    "python",
                    "-m",
                    "app.ingest.worker",
                )
                deadline = time.monotonic() + 120
                while True:
                    state = client.get(f"/jobs/{job}", headers=headers).json()
                    if state["status"] == "done":
                        break
                    if state["status"] == "dead" or time.monotonic() > deadline:
                        raise RuntimeError("Worker did not complete the test ingestion")
                    time.sleep(1)
                query = client.post(
                    "/query", headers=headers, json={"question": "Invoice 00123", "top_k": 5}
                )
                query.raise_for_status()
                assert any("007.00" in text for text in query.json()["contexts"])
                assert client.get("/metrics", headers=headers).status_code == 403
                docker("restart", api)
                port = docker("port", api, "8443/tcp").rsplit(":", 1)[1]
                client.base_url = f"https://localhost:{port}"
                ready = False
                for _ in range(180):
                    try:
                        if client.get("/readyz").status_code == 200:
                            ready = True
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                if not ready:
                    raise RuntimeError("Restart failed: " + docker("logs", api)[-3000:])
                query = client.post("/query", headers=headers, json={"question": "Invoice 00123"})
                query.raise_for_status()
                assert any("007.00" in text for text in query.json()["contexts"])
            with psycopg2.connect(cfg.postgres_dsn) as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=%s",
                    (role,),
                )
                assert cursor.fetchone() == (False, False, False)
                cursor.execute("SELECT has_schema_privilege(%s,%s,'CREATE')", (role, schema))
                assert cursor.fetchone() == (False,)
            assert docker("exec", api, "id", "-u") == "10001"
            print(
                json.dumps(
                    {
                        "tls_verified": True,
                        "nonroot_uid": 10001,
                        "database_superuser": False,
                        "runtime_schema_create": False,
                        "ingestion_completed": True,
                        "retrieval_after_restart": True,
                        "tenant_metrics_denied": True,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                )
            )
        finally:
            for name in (api, worker):
                subprocess.run(["docker", "rm", "--force", name], capture_output=True)
            for volume in (blob_volume, model_volume):
                subprocess.run(["docker", "volume", "rm", volume], capture_output=True)
            with psycopg2.connect(cfg.postgres_dsn) as connection, connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("REASSIGN OWNED BY {} TO {}").format(
                        sql.Identifier(role), sql.Identifier(params["user"])
                    )
                )
                cursor.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                cursor.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


if __name__ == "__main__":
    main()
