import json
import os
import select
import subprocess
import sys
import time

from app.shared.adapters.postgres.db import transaction


def test_worker_and_operator_scripts_use_disposable_storage(storage_settings):

    env = dict(
        os.environ,
        DATABASE_URL=storage_settings.database_url,
        REDIS_URL=storage_settings.redis_url,
        CACHE_INDEX_NAME=storage_settings.cache_index_name,
        DATA_DIR=str(storage_settings.data_dir),
        RERANKER_PROVIDER="none",
        EMBEDDING_PROVIDER="minilm",
        EMBEDDING_DIM="384",
        LITELLM_BASE_URL="",
        LITELLM_API_KEY="",
    )
    initialized = subprocess.run(
        [sys.executable, "-m", "scripts.init_database", "--schema"],
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert initialized.returncode == 0, initialized.stderr.decode()
    with transaction(storage_settings.database_url) as cur:
        cur.execute(
            "SELECT to_regclass('documents') AS documents, to_regclass('embedding_profile') AS profile"
        )
        assert all(cur.fetchone().values())
    seeded = subprocess.run(
        [sys.executable, "-m", "scripts.seed", "smoke", "smoke@example.test"],
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert seeded.returncode == 0
    with transaction(storage_settings.database_url) as cur:
        cur.execute("SELECT id FROM tenants WHERE name='smoke'")
        tenant_id = cur.fetchone()["id"]
    visibility = subprocess.run(
        [sys.executable, "-m", "scripts.set_visibility", tenant_id, "tenant", "--all", "--dry-run"],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert visibility.returncode == 0, visibility.stderr.decode()
    worker = subprocess.Popen(
        [sys.executable, "-m", "app.ingest.worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        deadline = time.monotonic() + 45
        started = False
        startup_lines = []
        while time.monotonic() < deadline and worker.poll() is None:
            if select.select([worker.stderr], [], [], 1)[0]:
                line = worker.stderr.readline()
                startup_lines.append(line.decode(errors="replace"))
                try:
                    started = json.loads(line).get("event") == "worker_start"
                except (ValueError, AttributeError):
                    continue
                if started:
                    break
        assert started, "Worker did not finish startup:\n" + "".join(startup_lines)
        assert worker.poll() is None
    finally:
        worker.terminate()
        worker.wait(timeout=10)
        worker.stderr.close()
    assert worker.returncode == 0
