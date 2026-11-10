import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from app.api.app import create_app
from app.shared import logger
from app.shared.gateway.client import LiteLLMClient


def test_cost_totals_nested_async_and_concurrent_isolation(caplog):
    caplog.set_level(logging.INFO)

    def run(cost):
        with logger.stage("retrieval", pipeline=True):
            with logger.stage("embedding"):
                logger.record_usage(
                    {"usage": {"prompt_tokens": 3}},
                    {"x-litellm-response-cost": str(cost)},
                    "embed",
                    1,
                )

            async def vision():
                logger.record_usage({"usage": {"completion_tokens": 5}}, {}, "vision", 2)

            asyncio.run(vision())

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, [0.1, 0.2]))
    totals = [r for r in caplog.records if getattr(r, "event", "") == "pipeline_done"]
    assert sorted(r.known_cost_usd for r in totals) == [0.1, 0.2]
    assert all(
        r.llm_calls == 2 and r.total_cost_usd is None and r.cost_unknown_calls == 1 for r in totals
    )
    assert all(r.input_tokens == 3 and r.output_tokens == 5 for r in totals)
    assert logger._usage.get() is None


def test_failure_reports_elapsed_and_resets_context(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(ValueError), logger.stage("ingestion", pipeline=True, job_id="job"):
        raise ValueError("failed")
    record = next(r for r in caplog.records if getattr(r, "event", "") == "pipeline_done")
    assert record.status == "error" and record.duration_ms >= 0
    assert logger._usage.get() is None
    assert "job_id" not in logger.correlation()


def test_api_headers_and_authenticated_request_log(container, tenant, monkeypatch):
    records = []
    original = logger.TextFormatter.format

    def capture(self, record):
        formatted = original(self, record)
        records.append(record.__dict__.copy())
        return formatted

    monkeypatch.setattr(logger.TextFormatter, "format", capture)
    with TestClient(create_app(container)) as client:
        response = client.get(
            "/documents",
            headers={
                "Authorization": f"Bearer {tenant['admin_token']}",
                "X-Request-ID": "request-1",
                "X-Session-ID": "session-1",
                "X-Tenant-ID": "untrusted",
                "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            },
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-1"
    assert response.headers["X-Session-ID"] == "session-1"
    event = next(r for r in records if r.get("event") == "request_end")
    assert event["tenant_id"] == tenant["id"] and event["user_id"] == tenant["admin_id"]
    assert "trace_id" not in event
    assert event["session_id"] == "session-1" and event["request_id"] == "request-1"


def test_gateway_evidence_logs_only_the_count_and_cost(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    evidence = [{"source_id": "1", "text": "完整 passage\n" * 10000}]
    seen = []

    async def request(self, url, key, payload, timeout):
        seen.append(payload)
        return httpx.Response(
            200,
            headers={"x-litellm-response-cost": "0.012"},
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            },
        )

    monkeypatch.setattr(LiteLLMClient, "_request", request)
    gateway = LiteLLMClient("http://gateway.test", "secret", "vision")
    with logger.stage("retrieval", pipeline=True):
        gateway.chat([{"role": "user", "content": json.dumps({"evidence": evidence})}], "model")
    record = next(r for r in caplog.records if getattr(r, "event", "") == "llm_evidence_prepared")
    assert record.chunks == 1
    assert evidence[0]["text"] not in logger.TextFormatter().format(record)
    assert json.loads(seen[0]["messages"][0]["content"])["evidence"] == evidence
    total = next(r for r in caplog.records if getattr(r, "event", "") == "pipeline_done")
    assert total.total_cost_usd == 0.012 and total.llm_calls == 1


def test_real_otlp_transport_exports_request_and_ingestion_boundaries_only():
    received = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        logger.configure_logging(
            settings=SimpleNamespace(
                app_env="test",
                otel_exporter_otlp_protocol="http/protobuf",
                otel_traces_sampler_arg="1.0",
                otel_exporter_otlp_endpoint=f"http://127.0.0.1:{server.server_port}",
                otel_exporter_otlp_headers="",
            )
        )
        with logger.log_context(
            request_id="request", session_id="session", tenant_id="tenant", user_id="user"
        ):
            with logger.stage("request", emit=False, trace_span=True, allow_root=True):
                queued = logger.correlation()
            with logger.stage(
                "ingestion", parent=queued, pipeline=True, trace_span=True, allow_root=True
            ):
                with logger.stage("embedding"):
                    logging.getLogger("app.test").info("indexed", extra={"chunks": 1})
        logger.shutdown_logging()
        logs = [
            ExportLogsServiceRequest.FromString(body)
            for path, body in received
            if path == "/v1/logs"
        ]
        traces = [
            ExportTraceServiceRequest.FromString(body)
            for path, body in received
            if path == "/v1/traces"
        ]
        records = [
            r
            for batch in logs
            for resource in batch.resource_logs
            for scope in resource.scope_logs
            for r in scope.log_records
        ]
        record = next(r for r in records if "indexed" in r.body.string_value)
        assert record.body.string_value == "indexed"
        attributes = {a.key: a.value.string_value for a in record.attributes}
        assert attributes["request_id"] == "request" and attributes["session_id"] == "session"
        assert attributes["tenant_id"] == "tenant" and attributes["user_id"] == "user"
        assert record.trace_id.hex() == attributes["trace_id"] and len(attributes["trace_id"]) == 32
        assert record.span_id.hex() == attributes["span_id"] and len(attributes["span_id"]) == 16
        spans = {
            s.name: s
            for batch in traces
            for resource in batch.resource_spans
            for scope in resource.scope_spans
            for s in scope.spans
        }
        assert set(spans) == {"request", "ingestion", "embedding"}
        assert spans["ingestion"].parent_span_id == spans["request"].span_id
        assert spans["ingestion"].trace_id == spans["request"].trace_id
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
        server.shutdown()
        server.server_close()
        thread.join()
        logger._traces = logger.TracerProvider()


def test_otlp_metrics_export_uses_bounded_http_dimensions():
    received = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        logger.configure_logging(
            settings=SimpleNamespace(
                app_env="test",
                otel_exporter_otlp_protocol="http/protobuf",
                otel_exporter_otlp_endpoint=f"http://127.0.0.1:{server.server_port}",
                otel_exporter_otlp_headers="",
                otel_metric_export_interval=60000,
            )
        )
        logger.record_http_request("POST", "/ask", 200, 0.25)
        logger.shutdown_logging()

        batches = [
            ExportMetricsServiceRequest.FromString(body)
            for path, body in received
            if path == "/v1/metrics"
        ]
        metrics = [
            metric
            for batch in batches
            for resource in batch.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        ]
        names = {metric.name for metric in metrics}
        assert {"http.server.request.count", "http.server.request.duration"} <= names
        count = next(metric for metric in metrics if metric.name == "http.server.request.count")
        attrs = {item.key: item.value for item in count.sum.data_points[0].attributes}
        assert attrs["http.request.method"].string_value == "POST"
        assert attrs["http.route"].string_value == "/ask"
        assert attrs["http.response.status_code"].int_value == 200
        assert "user_id" not in attrs and "request_id" not in attrs
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
        server.shutdown()
        server.server_close()
        thread.join()
        logger._traces = logger.TracerProvider()
