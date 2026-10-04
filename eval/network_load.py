from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import numpy as np
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.adapters.request_gate import RedisRequestGate
from app.shared.container import build_container
from app.shared.ports.vector_store import VectorPoint
from scripts.disposable_storage import evaluation_settings

log = logging.getLogger(__name__)


class PartitionProxy:
    def __init__(self, target_host: str, target_port: int):
        self.target = (target_host, target_port)
        self.partitioned = threading.Event()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.connections = set()
        self.server = asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(5)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _start(self):
        return await asyncio.start_server(self._connect, "127.0.0.1", 0)

    async def _connect(self, reader, writer):
        remote = None
        tasks = []
        self.connections.add(asyncio.current_task())
        try:
            upstream, remote = await asyncio.open_connection(*self.target)

            async def forward(source, destination):
                while data := await source.read(65536):
                    while self.partitioned.is_set():
                        await asyncio.sleep(0.02)
                    destination.write(data)
                    await destination.drain()

            tasks = [
                asyncio.create_task(forward(reader, remote)),
                asyncio.create_task(forward(upstream, writer)),
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            if remote is not None:
                remote.close()
            self.connections.discard(asyncio.current_task())

    def close(self):
        async def shutdown():
            self.server.close()
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.server.wait_closed()

        asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


def exercise(seconds: float = 60, workers: int = 8) -> dict:
    with evaluation_settings() as settings:
        container = build_container(settings.model_copy(update={"reranker_provider": "none"}))
        tenant = container.metadata.create_tenant("load-acceptance")
        user = container.metadata.create_user(tenant, "load@test.example", "admin", "load-token")
        vector = container.embedder.embed_query("Invoice 00123")
        container.vectors.upsert(
            [
                VectorPoint(
                    "load-evidence",
                    tenant,
                    vector,
                    {
                        "_id": "load-document",
                        "user_id": user,
                        "visibility": "tenant",
                        "content": "Invoice 00123 amount 007.00",
                    },
                )
            ]
        )
        address = urlsplit(settings.redis_url)
        if address.password or address.scheme != "redis":
            raise ValueError(
                "Partition probe requires a disposable unauthenticated redis:// service"
            )
        proxy = PartitionProxy(address.hostname, address.port or 6379)
        original_gate = container.request_gate
        gate = RedisRequestGate(
            f"redis://127.0.0.1:{proxy.port}",
            settings.cache_index_name,
            global_concurrency=workers + 2,
            tenant_concurrency=workers + 2,
            user_concurrency=workers + 2,
            tenant_per_minute=100000,
            user_per_minute=100000,
        )
        container.request_gate = gate
        headers = {"Authorization": "Bearer load-token"}
        try:
            with TestClient(create_app(container)) as client:
                assert (
                    client.post(
                        "/query", headers=headers, json={"question": "Invoice 00123"}
                    ).status_code
                    == 200
                )
                proxy.partitioned.set()
                start = time.monotonic()
                failed = client.post("/query", headers=headers, json={"question": "Invoice 00123"})
                failure_ms = (time.monotonic() - start) * 1000
                assert failed.status_code == 503 and failure_ms < 10000
                proxy.partitioned.clear()
                recovered = client.post(
                    "/query", headers=headers, json={"question": "Invoice 00123"}
                )
                assert recovered.status_code == 200
                deadline = time.monotonic() + seconds

                def run(index):
                    observations = []
                    while time.monotonic() < deadline:
                        start = time.monotonic()
                        response = client.post(
                            "/query",
                            headers=headers,
                            json={"question": "Invoice 00123", "top_k": 1},
                        )
                        observations.append(
                            (
                                response.status_code,
                                (time.monotonic() - start) * 1000,
                                response.status_code == 200
                                and response.json()["contexts"] == ["Invoice 00123 amount 007.00"],
                            )
                        )
                    return observations

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    observations = [
                        item for batch in pool.map(run, range(workers)) for item in batch
                    ]
                successful = sum(status == 200 and correct for status, _, correct in observations)
                timings = [duration for _, duration, _ in observations]
                result = {
                    "duration_seconds": seconds,
                    "workers": workers,
                    "requests": len(observations),
                    "successful_correct": successful,
                    "partition_status": failed.status_code,
                    "status_counts": dict(Counter(status for status, _, _ in observations)),
                    "partition_failure_ms": round(failure_ms, 2),
                    "recovery_status": recovered.status_code,
                    "p50_ms": round(float(np.percentile(timings, 50)), 2),
                    "p95_ms": round(float(np.percentile(timings, 95)), 2),
                    "p99_ms": round(float(np.percentile(timings, 99)), 2),
                }
                if successful != len(observations) or result["p95_ms"] >= 5000:
                    raise RuntimeError(json.dumps(result))
                return result
        finally:
            proxy.close()
            gate.close()
            original_gate.close()
            container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.seconds <= 86400 or not 1 <= arguments.workers <= 32:
        parser.error("seconds must be 1..86400 and workers 1..32")
    logging.disable(logging.CRITICAL)
    print(json.dumps(exercise(arguments.seconds, arguments.workers), indent=2))
