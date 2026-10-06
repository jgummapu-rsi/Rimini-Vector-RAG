import asyncio
import io
import json
import multiprocessing
import os
import struct
import time
import zipfile

import httpx
import pytest
from PIL import Image

from app.ingest.pipeline.parse_process import _receive, extract_bounded
from app.ingest.pipeline.safety import UnsafeContentError, check_archive_expansion
from app.ingest.worker import _handle_job_failure
from app.shared.config import Settings
from app.shared.domain.models import Document, Job
from app.shared.gateway.client import LiteLLMClient, VisionText
from app.shared.ids import new_object_id


def _picture():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_vision_executes_in_parent_and_survives_retry(container, tenant):

    doc = Document(
        new_object_id(),
        tenant["id"],
        tenant["admin_id"],
        "image",
        "unused",
        "contenthash",
        "image/png",
        "scan.png",
        "private",
        [],
    )
    job = Job(new_object_id(), doc.id, tenant["id"], "parse", "running", 0)
    container.metadata.create_document_with_job(doc, job)
    calls = []

    async def transcribe(*args, **kwargs):
        calls.append(os.getpid())
        return "Invoice 00123 total 007.00 USD"

    container.gateway.vision_bounded = transcribe

    def load(key):
        return container.metadata.get_extraction_artifact(job.id, doc.content_sha256, key)

    def save(key, artifact):
        return container.metadata.put_extraction_artifact(job.id, doc.content_sha256, key, artifact)

    first, first_summary = extract_bounded(
        "scan.png", _picture(), container.gateway, container.settings, load, save
    )
    second, second_summary = extract_bounded(
        "scan.png", _picture(), container.gateway, container.settings, load, save
    )
    assert calls == [os.getpid()]
    assert first == second
    assert first[0].text == "Invoice 00123 total 007.00 USD"
    assert first_summary["vision_calls"] == 1
    assert second_summary["extraction_reused"] is True
    assert second_summary["vision_calls"] == 0


def test_vision_failure_reaps_native_child(container):
    async def transcribe(*args, **kwargs):
        raise TimeoutError("vision deadline")

    container.gateway.vision_bounded = transcribe
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(TimeoutError, match="vision deadline"):
        extract_bounded(
            "scan.png",
            _picture(),
            container.gateway,
            container.settings,
            lambda key: None,
            lambda key, artifact: None,
        )
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_office_expansion_budget_is_checked_before_loading():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 100000)
    with pytest.raises(UnsafeContentError, match="expansion-ratio"):
        check_archive_expansion(buffer.getvalue(), "bomb.docx", Settings(_env_file=None))


def test_native_memory_limit_prevents_large_evidence_allocation():
    cfg = Settings(_env_file=None, parse_memory_mb=1)
    with pytest.raises(UnsafeContentError):
        extract_bounded(
            "large.txt",
            b"large text " * 10000,
            None,
            cfg,
            lambda key: None,
            lambda key, artifact: None,
        )


def test_malformed_content_is_dead_lettered_without_repeated_paid_attempts(container, tenant):

    doc = Document(
        new_object_id(),
        tenant["id"],
        tenant["admin_id"],
        "pdf",
        "unused",
        "malformed",
        "application/pdf",
        "bad.pdf",
        "private",
        [],
    )
    job = Job(new_object_id(), doc.id, tenant["id"], "parse", "queued", 0)
    container.metadata.create_document_with_job(doc, job)
    claimed = container.queue.claim_next()
    _handle_job_failure(container, claimed, UnsafeContentError("Malformed file"), 5)
    stored = container.metadata.get_job(tenant["id"], job.id)
    assert stored.status == "dead"
    assert stored.attempts == 1
    assert container.queue.claim_next() is None


def test_vision_network_request_is_cancelled_on_total_deadline(monkeypatch):
    cancelled = []
    request_payloads = []

    async def handle(request):
        request_payloads.append(json.loads(request.content))
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        return httpx.Response(200)

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: factory(transport=httpx.MockTransport(handle), **kwargs),
    )
    gateway = LiteLLMClient("https://gateway.test", "test", "vision")
    with pytest.raises(TimeoutError):
        asyncio.run(
            gateway.vision_bounded(
                _picture(), "transcribe", "image/png", timeout_seconds=0.05, max_tokens=32
            )
        )
    assert cancelled == [True]
    assert request_payloads[0]["max_tokens"] == 32


def test_partial_native_ipc_frame_cannot_bypass_deadline():

    parent, child = multiprocessing.Pipe()
    try:
        os.write(child.fileno(), struct.pack("!I", 100) + b"{")
        with pytest.raises(TimeoutError, match="IPC"):
            _receive(parent, time.monotonic() + 0.05)
    finally:
        parent.close()
        child.close()


def test_many_short_vision_responses_use_actual_tokens_and_reuse_on_retry():
    frames = [Image.new("RGB", (8, 8), (index, 0, 0)) for index in range(65)]
    image = io.BytesIO()
    frames[0].save(image, format="TIFF", save_all=True, append_images=frames[1:])
    artifacts = {}

    class Gateway:
        vision_model = "test-vision"
        calls = 0
        fail_at = 40

        async def vision_bounded(self, *args, **kwargs):
            self.calls += 1
            if self.calls == self.fail_at:
                raise TimeoutError("temporary gateway failure")
            assert kwargs["max_tokens"] == 8192
            return VisionText("Invoice number 00123", 10)

    gateway = Gateway()
    cfg = Settings(_env_file=None)
    args = ("frames.tiff", image.getvalue(), gateway, cfg, artifacts.get, artifacts.__setitem__)
    with pytest.raises(TimeoutError, match="temporary"):
        extract_bounded(*args)
    gateway.fail_at = None
    elements, summary = extract_bounded(*args)
    assert len(elements) == 65
    assert summary["vision_reused"] == 39
    assert summary["vision_calls"] == 26
    assert summary["vision_completion_tokens"] == 260


def test_vision_usage_still_enforces_total_completion_budget():
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(
        image, format="TIFF", save_all=True, append_images=[Image.new("RGB", (8, 8), "black")]
    )

    class Gateway:
        vision_model = "test-vision"
        calls = 0

        async def vision_bounded(self, *args, **kwargs):
            self.calls += 1
            assert kwargs["max_tokens"] == 20
            return VisionText("Evidence", 20)

    gateway = Gateway()
    with pytest.raises(UnsafeContentError, match="completion-token budget"):
        extract_bounded(
            "frames.tiff",
            image.getvalue(),
            gateway,
            Settings(_env_file=None, max_vision_tokens=20),
            lambda key: None,
            lambda key, value: None,
        )
    assert gateway.calls == 1


@pytest.mark.parametrize("usage,expected", [({"completion_tokens": 7}, 7), ({}, 32)])
def test_gateway_vision_returns_usage_with_conservative_fallback(monkeypatch, usage, expected):
    async def handle(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "Invoice 00123"}}],
                "usage": usage,
            },
        )

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: factory(transport=httpx.MockTransport(handle), **kwargs),
    )
    gateway = LiteLLMClient("https://gateway.test", "test", "vision")
    result = asyncio.run(
        gateway.vision_bounded(
            _picture(),
            "transcribe",
            "image/png",
            timeout_seconds=5,
            max_tokens=32,
        )
    )
    assert result == "Invoice 00123"
    assert result.completion_tokens == expected
