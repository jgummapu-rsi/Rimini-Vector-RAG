"""Harness checks without application storage or an API test-client substitute."""

import asyncio
import os
import socket
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.shared.config import Settings
from eval import multi_replica_load as probe


def test_child_environment_overrides_application_and_mounted_secrets(monkeypatch, tmp_path):

    monkeypatch.setenv("DATABASE_URL", "application-database")
    monkeypatch.setenv("REDIS_URL", "application-redis")
    monkeypatch.setenv("DATABASE_URL_FILE", str(tmp_path / "absent-secret"))

    cfg = Settings.model_construct(
        database_url="disposable-database",
        redis_url="redis://disposable",
        cache_index_name="unique-index",
        data_dir=tmp_path,
    )
    environment = probe.child_environment(cfg)
    assert environment["DATABASE_URL"] == cfg.database_url
    assert environment["REDIS_URL"] == cfg.redis_url
    assert environment["CACHE_INDEX_NAME"] == cfg.cache_index_name
    assert environment["DATA_DIR"] == str(tmp_path)
    assert "DATABASE_URL_FILE" not in environment
    assert environment["PYTHONPATH"] == str(probe.ROOT)
    assert environment["LITELLM_API_KEY"] == ""
    assert os.environ["DATABASE_URL"] == "application-database"


def test_launch_passes_bound_socket_and_env_only(monkeypatch, tmp_path):
    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)

        with socket.fromfd(kwargs["pass_fds"][0], socket.AF_INET, socket.SOCK_STREAM) as sock:
            captured["address"] = sock.getsockname()
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    settings = SimpleNamespace(
        data_dir=tmp_path,
        model_dump=lambda **_: {
            "database_url": "secret-dsn",
            "redis_url": "secret-redis",
        },
    )
    processes = []
    url = probe.launch(settings, processes)
    assert url == f"http://127.0.0.1:{captured['address'][1]}"
    assert processes[0].pid == 123
    assert "app.api.app:app" in captured["command"]
    assert "secret" not in " ".join(captured["command"])
    assert captured["env"]["DATABASE_URL"] == "secret-dsn"
    assert captured["cwd"] == tmp_path
    assert captured["stdout"] == captured["stderr"] == subprocess.DEVNULL


def test_reporting_keeps_failures_and_wrong_evidence_in_denominators():
    observations = [
        probe.Observation(0, "baseline", 200, True, 10),
        probe.Observation(1, "baseline", 200, False, 20, "incorrect_evidence"),
        probe.Observation(0, "outage", 503, False, 30, "http_503"),
        probe.Observation(1, "outage", None, False, 40, "ReadTimeout"),
    ]
    report = probe.summarize(observations)
    assert report["http_success"] == {"numerator": 2, "denominator": 4}
    assert report["correct_evidence"] == {"numerator": 1, "denominator": 4}
    assert report["correct_given_http_success"] == {"numerator": 1, "denominator": 2}
    assert report["errors"] == {"incorrect_evidence": 1, "http_503": 1, "ReadTimeout": 1}
    assert report["p50_ms"] == 25
    assert report["p99_ms"] > 39
    assert probe.summarize([])["p95_ms"] is None


def test_evidence_checks_identity_and_citation_not_only_http_status():
    evidence = probe.Evidence("secret", "expected", "chunk-a", "document-a")
    body = {
        "question": probe.QUESTION,
        "contexts": ["expected"],
        "chunk_ids": ["chunk-a"],
        "citations": [{"chunk_id": "chunk-a", "document_id": "document-a", "snippet": "expected"}],
    }
    assert probe.correct_evidence(body, evidence)
    assert not probe.correct_evidence(body, replace(evidence, document_id="other-tenant"))
    assert not probe.correct_evidence(body, replace(evidence, content="other-tenant"))
    for invalid in ([], None, {}, {**body, "citations": [None]}, {**body, "chunk_ids": []}):
        assert not probe.correct_evidence(invalid, evidence)


def test_eight_clients_each_alternate_replicas(monkeypatch):
    seen = {}

    async def request(client, url, evidence, replica, phase):
        if phase != "post_run":
            seen.setdefault(id(client), []).append(replica)
        await asyncio.sleep(0.005)
        return probe.Observation(replica, phase, 200, True, 5)

    monkeypatch.setattr(probe, "request", request)
    result = asyncio.run(probe.load(["replica-0", "replica-1"], [None, None], 0.08, 0, None))
    assert len(seen) == 8
    for replicas in seen.values():
        assert len(replicas) >= 2
        assert all(a != b for a, b in zip(replicas, replicas[1:], strict=False))
    assert result["post_run_probes"]["requests"] == 4
    assert result["overall"]["requests"] == sum(map(len, seen.values()))


def test_request_reports_transport_errors_without_exception_credentials():
    def fail(request):
        raise httpx.ReadTimeout("secret-dsn", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            return await probe.request(
                client, "http://replica", probe.Evidence("token", "", "", ""), 0, "outage"
            )

    observation = asyncio.run(run())
    assert observation.status is None
    assert observation.error == "ReadTimeout"
    assert "secret-dsn" not in repr(observation)


def test_cleanup_reaps_real_child():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        probe.stop_processes([process])
        assert process.poll() is not None
        probe.stop_processes([process])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize(
    "seconds,outage", [(0, 0), (float("nan"), 0), (15, -1), (15, 14), (15, float("inf"))]
)
def test_invalid_durations_rejected_before_storage(seconds, outage, monkeypatch):
    def forbidden():
        pytest.fail("invalid arguments must not open storage")

    monkeypatch.setattr(probe, "evaluation_settings", forbidden)
    with pytest.raises(ValueError):
        probe.exercise(seconds, outage)
