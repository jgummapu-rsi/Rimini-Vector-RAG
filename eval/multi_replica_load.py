"""Local two-process retrieval probe; requires EVAL_DATABASE_URL/EVAL_REDIS_URL.

Run with ``python -m eval.multi_replica_load --seconds 60``. An optional
``--outage-seconds 10`` pauses Redis TCP forwarding mid-run. This is a small,
closed-loop synthetic retrieval workload, not a production capacity benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
import numpy as np

from app.shared import container as composition
from app.shared.ports.vector_store import VectorPoint
from eval.network_load import PartitionProxy
from scripts.disposable_storage import evaluation_settings

CLIENTS = 8
REPLICAS = 2
QUESTION = "Invoice 00123"
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Evidence:
    token: str
    content: str
    chunk_id: str
    document_id: str


@dataclass(frozen=True)
class Observation:
    replica: int
    phase: str
    status: int | None
    correct: bool
    milliseconds: float
    error: str | None = None


def child_environment(settings) -> dict[str, str]:
    """Pass credentials only via env; override every application setting.

    The child also runs in the disposable data directory, away from repo .env.
    Mounted-secret selectors must be removed because they outrank env values.
    """
    environment = os.environ.copy()
    for name in list(environment):
        if name.upper().endswith("_FILE") or name.upper() in {
            "METADATA_BACKEND",
            "VECTOR_BACKEND",
            "QUEUE_BACKEND",
            "BLOB_BACKEND",
        }:
            environment.pop(name)
    for name, value in settings.model_dump(mode="json").items():
        environment[name.upper()] = (
            json.dumps(value)
            if isinstance(value, (list, dict, bool))
            else ""
            if value is None
            else str(value)
        )
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def seed(settings) -> list[Evidence]:
    container = composition.build_container(settings)
    try:
        vector = container.embedder.embed_query(QUESTION)
        evidence = []
        for number in range(2):
            identity = uuid4().hex
            tenant = container.metadata.create_tenant("replica-load-" + identity)
            token = secrets.token_urlsafe(32)
            user = container.metadata.create_user(
                tenant,
                identity + "@example.test",
                "viewer",
                token,
            )
            item = Evidence(
                token,
                f"Invoice 00123 amount {number + 7:03}.00 USD.",
                "chunk-" + identity,
                "document-" + identity,
            )
            container.vectors.upsert(
                [
                    VectorPoint(
                        item.chunk_id,
                        tenant,
                        vector,
                        {
                            "_id": item.document_id,
                            "user_id": user,
                            "visibility": "tenant",
                            "scope": "tenant",
                            "content": item.content,
                        },
                    )
                ]
            )
            evidence.append(item)
        return evidence
    finally:
        container.close()


def stop_processes(processes) -> None:

    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def launch(settings, processes) -> str:

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.api.app:app",
                "--fd",
                str(listener.fileno()),
                "--workers",
                "1",
                "--no-access-log",
                "--log-level",
                "error",
                "--timeout-graceful-shutdown",
                "5",
            ],
            pass_fds=(listener.fileno(),),
            env=child_environment(settings),
            cwd=settings.data_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(process)
    return f"http://127.0.0.1:{port}"


def wait_ready(url: str, process, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"API exited before readiness (exit {process.returncode})")
            try:
                response = client.get(url + "/readyz")
                if response.status_code == 200 and response.json() == {"status": "ready"}:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.2)
    raise RuntimeError("API readiness timed out")


def correct_evidence(body, evidence: Evidence) -> bool:
    if not isinstance(body, dict):
        return False
    citations = body.get("citations")
    return (
        body.get("question") == QUESTION
        and body.get("contexts") == [evidence.content]
        and body.get("chunk_ids") == [evidence.chunk_id]
        and isinstance(citations, list)
        and len(citations) == 1
        and isinstance(citations[0], dict)
        and citations[0].get("chunk_id") == evidence.chunk_id
        and citations[0].get("document_id") == evidence.document_id
        and citations[0].get("snippet") == evidence.content
    )


async def request(client, url, evidence, replica, phase) -> Observation:
    started = time.monotonic()
    status, correct, error = None, False, None
    try:
        response = await client.post(
            url + "/query",
            headers={"Authorization": "Bearer " + evidence.token},
            json={"question": QUESTION, "top_k": 1},
        )
        status = response.status_code
        if status == 200:
            try:
                correct = correct_evidence(response.json(), evidence)
                error = None if correct else "incorrect_evidence"
            except ValueError:
                error = "invalid_json"
        else:
            error = f"http_{status}"
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    return Observation(replica, phase, status, correct, (time.monotonic() - started) * 1000, error)


def summarize(observations) -> dict:
    count = len(observations)
    successful = sum(item.status == 200 for item in observations)
    correct = sum(item.correct for item in observations)
    timings = [item.milliseconds for item in observations]
    return {
        "requests": count,
        "http_success": {"numerator": successful, "denominator": count},
        "correct_evidence": {"numerator": correct, "denominator": count},
        "correct_given_http_success": {"numerator": correct, "denominator": successful},
        "status_counts": dict(
            Counter(
                str(item.status) if item.status is not None else "transport_error"
                for item in observations
            )
        ),
        "errors": dict(Counter(item.error for item in observations if item.error)),
        **{
            f"p{p}_ms": round(float(np.percentile(timings, p)), 2) if timings else None
            for p in (50, 95, 99)
        },
    }


async def load(urls, evidence, seconds, outage_seconds, proxy) -> dict:
    async with AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(
                httpx.AsyncClient(
                    timeout=15,
                    trust_env=False,
                )
            )
            for _ in range(CLIENTS)
        ]
        started = time.monotonic()
        deadline = started + seconds
        phase = "baseline"
        fault_timing = {}

        async def fault():
            nonlocal phase
            if not outage_seconds:
                return
            await asyncio.sleep((seconds - outage_seconds) / 2)
            proxy.partitioned.set()
            phase = "outage"
            fault_started = time.monotonic()
            fault_timing["started_at_seconds"] = round(fault_started - started, 3)
            try:
                await asyncio.sleep(outage_seconds)
            finally:
                proxy.partitioned.clear()
                phase = "recovery"
                fault_timing["duration_seconds"] = round(time.monotonic() - fault_started, 3)

        async def run(index):
            observations = []
            sequence = 0
            while time.monotonic() < deadline:
                replica = (index + sequence) % REPLICAS
                observations.append(
                    await request(
                        clients[index],
                        urls[replica],
                        evidence[index % len(evidence)],
                        replica,
                        phase,
                    )
                )
                sequence += 1
            return observations

        fault_task = asyncio.create_task(fault())
        workers = [asyncio.create_task(run(index)) for index in range(CLIENTS)]
        try:
            batches = await asyncio.gather(*workers)
            await fault_task
        finally:
            for task in [*workers, fault_task]:
                task.cancel()
            await asyncio.gather(*workers, fault_task, return_exceptions=True)
        elapsed = time.monotonic() - started
        observations = [item for batch in batches for item in batch]

        probes = [
            await request(clients[0], url, item, replica, "post_run")
            for replica, url in enumerate(urls)
            for item in evidence
        ]
    return {
        "requested_duration_seconds": seconds,
        "duration_seconds": round(elapsed, 3),
        "drain_seconds": round(max(0, elapsed - seconds), 3),
        "replicas": REPLICAS,
        "clients": CLIENTS,
        "outage_requested_seconds": outage_seconds,
        "redis_fault": fault_timing,
        "overall": summarize(observations),
        "by_replica": {
            str(replica): summarize([o for o in observations if o.replica == replica])
            for replica in range(REPLICAS)
        },
        "by_phase": {
            name: summarize([o for o in observations if o.phase == name])
            for name in ("baseline", "outage", "recovery")
        },
        "post_run_probes": summarize(probes),
        "limitations": [
            "Single-host POSIX, direct loopback HTTP; no load balancer or TLS.",
            "Eight closed-loop clients; two tenants, one synthetic vector each; no ingestion or generation.",
            "Real MiniLM; reranker disabled; admission limits raised for this probe.",
            "Redis admission and readiness exercised; semantic answer-cache hits are not exercised.",
            "Latency includes failures; phases label request start, so requests may cross fault boundaries.",
            "Startup and post-run probes excluded from load duration and request totals.",
        ],
    }


def exercise(seconds: float = 60, outage_seconds: float = 0) -> dict:
    if not math.isfinite(seconds) or not 1 <= seconds <= 86400:
        raise ValueError("seconds must be finite and in 1..86400")
    if (
        not math.isfinite(outage_seconds)
        or outage_seconds < 0
        or (outage_seconds and outage_seconds > seconds - 2)
    ):
        raise ValueError(
            "outage-seconds must be nonnegative and leave at least two fault-free seconds"
        )
    if os.name != "posix":
        raise RuntimeError("This probe requires POSIX inherited sockets")
    with evaluation_settings() as settings, ExitStack() as stack:
        settings = settings.model_copy(
            update={
                "embedding_provider": "minilm",
                "reranker_provider": "none",
                "rerank_min_score": -3.0,
                "litellm_base_url": "",
                "litellm_api_key": "",
                "platform_tenant_id": "",
                "request_global_concurrency": 32,
                "request_tenant_concurrency": 16,
                "request_user_concurrency": 16,
                "request_tenant_per_minute": 100000,
                "request_user_per_minute": 100000,
            }
        )
        evidence = seed(settings)
        proxy = None
        if outage_seconds:
            address = urlsplit(settings.redis_url)
            if address.scheme != "redis" or not address.hostname:
                raise ValueError("TCP fault probe requires redis://")
            proxy = PartitionProxy(address.hostname, address.port or 6379)
            stack.callback(proxy.close)

            credentials = address.netloc.rsplit("@", 1)[0] + "@" if "@" in address.netloc else ""
            redis_url = urlunsplit(address._replace(netloc=f"{credentials}127.0.0.1:{proxy.port}"))
            settings = settings.model_copy(update={"redis_url": redis_url})
        processes, urls = [], []
        stack.callback(stop_processes, processes)
        for _ in range(REPLICAS):
            urls.append(launch(settings, processes))
            wait_ready(urls[-1], processes[-1])
        result = asyncio.run(load(urls, evidence, seconds, outage_seconds, proxy))
        result["replica_pids"] = [process.pid for process in processes]
        result["replicas_alive_at_end"] = [process.poll() is None for process in processes]
    result["replicas_reaped"] = [process.poll() is not None for process in processes]
    result["disposable_storage_cleanup_completed"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--outage-seconds", type=float, default=0)
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = exercise(arguments.seconds, arguments.outage_seconds)
    except KeyboardInterrupt:
        print(json.dumps({"error": "interrupted"}))
        return 130
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "stage": "setup_or_execution"}))
        return 1
    print(json.dumps(result, indent=2))
    total = result["overall"]
    probes = result["post_run_probes"]
    return int(
        not all(result["replicas_alive_at_end"])
        or probes["correct_evidence"]["numerator"] != probes["requests"]
        or total["correct_evidence"]["numerator"] != total["http_success"]["numerator"]
        or (
            not arguments.outage_seconds and total["http_success"]["numerator"] != total["requests"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
