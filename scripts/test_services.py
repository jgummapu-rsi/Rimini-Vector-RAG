"""Disposable storage for plain pytest, including the centrally managed org gate.

Explicit TEST_DATABASE_URL/TEST_REDIS_URL take precedence. Otherwise start only
new, randomly named containers on loopback ephemeral ports and remove them on exit.
The controller exports URLs before xdist starts so workers share the services;
disposable_settings still creates an isolated database/index for every test.
"""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager
from uuid import uuid4

POSTGRES_IMAGE = "pgvector/pgvector:0.8.2-pg16"
REDIS_IMAGE = "redis/redis-stack-server:7.4.0-v8"


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-2000:]}")
    return result.stdout.strip()


def _wait_healthy(name: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = _docker("inspect", "--format", "{{.State.Health.Status}}", name)
        if status == "healthy":
            return
        if status == "unhealthy":
            raise RuntimeError(f"Test storage container {name} failed its health check")
        time.sleep(1)
    raise TimeoutError(f"Test storage container {name} did not become healthy")


@contextmanager
def test_services():
    database = os.environ.get("TEST_DATABASE_URL")
    cache = os.environ.get("TEST_REDIS_URL")
    if database and cache:
        yield
        return
    if database or cache:
        raise ValueError("Set both TEST_DATABASE_URL and TEST_REDIS_URL, or neither")

    identity = uuid4().hex
    postgres, redis = f"rag-pytest-pg-{identity}", f"rag-pytest-redis-{identity}"
    names = []
    previous = {}
    try:
        _docker("info", "--format", "{{.ServerVersion}}")
        for name, port, image, health, env in (
            (
                postgres,
                "5432",
                POSTGRES_IMAGE,
                "pg_isready -U postgres -d rag_test",
                ("-e", "POSTGRES_PASSWORD=rag-test-only", "-e", "POSTGRES_DB=rag_test"),
            ),
            (redis, "6379", REDIS_IMAGE, "redis-cli ping", ()),
        ):
            # Register before starting so partial starts are also cleaned up.
            names.append(name)
            _docker(
                "run",
                "--detach",
                "--name",
                name,
                "--publish",
                f"127.0.0.1::{port}",
                "--health-cmd",
                health,
                "--health-interval",
                "1s",
                "--health-timeout",
                "3s",
                "--health-retries",
                "60",
                *env,
                image,
            )
            _wait_healthy(name)
        pg_port = _docker("port", postgres, "5432/tcp").rsplit(":", 1)[1]
        redis_port = _docker("port", redis, "6379/tcp").rsplit(":", 1)[1]
        variables = {
            "TEST_DATABASE_URL": f"postgresql://postgres:rag-test-only@127.0.0.1:{pg_port}/rag_test",
            "TEST_REDIS_URL": f"redis://127.0.0.1:{redis_port}",
            "RESTORE_DRILL_TOOLS_CONTAINER": postgres,
        }
        for key, value in variables.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name in reversed(names):
            subprocess.run(
                ["docker", "rm", "--force", "--volumes", name],
                capture_output=True,
                timeout=30,
            )
