import logging
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from app.api.app import create_app
from app.api.ingest_routes import ingest, reprocess
from app.ingest.pipeline import runner
from app.shared import logger
from app.shared.domain.models import ChunkRecord, Document, Job
from app.shared.ids import new_object_id
from app.shared.observability import get_trace_request_type
from app.shared.ports.vector_store import VectorPoint


def _collect_spans(received):
    traces = [
        ExportTraceServiceRequest.FromString(body)
        for path, body in received
        if path == "/v1/traces"
    ]
    resources = []
    for batch in traces:
        for resource in batch.resource_spans:
            attrs = {a.key: a.value.string_value for a in resource.resource.attributes}
            spans = [s for scope in resource.scope_spans for s in scope.spans]
            resources.append((attrs, spans))
    return resources


def _span_attrs(span):
    attrs = {}
    for item in span.attributes:
        value = item.value
        kind = value.WhichOneof("value")
        if kind == "string_value":
            attrs[item.key] = value.string_value
        elif kind == "int_value":
            attrs[item.key] = value.int_value
        elif kind == "double_value":
            attrs[item.key] = value.double_value
        elif kind == "bool_value":
            attrs[item.key] = value.bool_value
        else:
            attrs[item.key] = ""
    return attrs


@contextmanager
def _otlp_receiver():
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
    try:
        yield received, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_tracing_contract_health_filtered_and_ask_single_trace(container, tenant):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with _otlp_receiver() as (received, port):
        try:
            container.settings.otel_service_name = "rimini-rag-api"
            container.settings.otel_exporter_otlp_protocol = "http/protobuf"
            container.settings.otel_traces_sampler_arg = "1.0"
            container.settings.otel_exporter_otlp_endpoint = f"http://127.0.0.1:{port}"
            container.settings.otel_exporter_otlp_headers = ""

            text = "The document says quarterly revenue increased."
            document_id = new_object_id()
            chunk_id = f"{document_id}:000001:000001"
            container.metadata.create_document(
                Document(
                    id=document_id,
                    tenant_id=tenant["id"],
                    owner_user_id=tenant["admin_id"],
                    source_type="text",
                    blob_path="unused",
                    content_sha256="0" * 64,
                    mime="text/plain",
                    filename="trace.txt",
                    visibility="private",
                    acl_user_ids=[],
                )
            )
            container.vectors.upsert(
                [
                    VectorPoint(
                        chunk_id,
                        tenant["id"],
                        container.embedder.embed_query(text),
                        {
                            "_id": document_id,
                            "user_id": tenant["admin_id"],
                            "visibility": "private",
                            "acl_user_ids": [],
                            "scope": "tenant",
                            "content": text,
                            "filename": "trace.txt",
                        },
                    )
                ]
            )
            container.metadata.replace_document_chunks(
                tenant["id"],
                document_id,
                [ChunkRecord(0, "text", "test", "test", 8, text, id=chunk_id, content_sha256="1" * 64)],
            )
            container.gateway.chat = lambda *args, **kwargs: (
                '{"status":"answered","answer":"Revenue increased [1]","source_ids":["1"]}'
            )

            app = create_app(container)
            with TestClient(app) as client:
                health = client.get("/healthz")
                assert health.status_code == 200

                docs = client.get(
                    "/documents",
                    headers={
                        "Authorization": f"Bearer {tenant['admin_token']}",
                        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
                    },
                )
                assert docs.status_code == 200

                ask = client.post(
                    "/ask",
                    headers={
                        "Authorization": f"Bearer {tenant['admin_token']}",
                        "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
                    },
                    json={
                        "question": "What is in the document?",
                        "top_k": 3,
                        "model": "gpt-6-sol",
                    },
                )
                assert ask.status_code == 200

            resources = _collect_spans(received)
            all_spans = [span for _, spans in resources for span in spans]
            span_names = [span.name for span in all_spans]
            by_trace = {}
            for span in all_spans:
                by_trace.setdefault(span.trace_id.hex(), []).append(span)

            assert "GET /documents" not in span_names
            assert "POST /ask" in span_names
            assert "retrieve_documents" in span_names
            assert "llm_generate" in span_names
            if container.reranker is None:
                assert "rerank_documents" not in span_names
            else:
                assert "rerank_documents" in span_names
            assert "GET /healthz" not in span_names

            ask_trace_ids = {
                span.trace_id
                for span in all_spans
                if span.name in {"POST /ask", "llm_generate"}
            }
            assert len(ask_trace_ids) == 1

            ask_trace = by_trace["0123456789abcdef0123456789abcdef"]
            ask_names = {span.name for span in ask_trace}
            assert {
                "POST /ask",
                "retrieval",
                "query_embedding",
                "query_routing",
                "retrieve_documents",
                "reranking",
                "context_expansion",
                "answer_generation",
                "context_packing",
                "llm_generate",
                "answer_validation",
            }.issubset(ask_names)
            ask_root = next(span for span in ask_trace if span.name == "POST /ask")
            local_span_ids = {span.span_id for span in ask_trace}
            assert len(ask_trace) > 10
            assert all(
                span is ask_root or span.parent_span_id in local_span_ids for span in ask_trace
            )
            root_attrs = _span_attrs(ask_root)
            assert root_attrs["http.route"] == "/ask"
            assert root_attrs["http.method"] == "POST"
            assert root_attrs["http.target"] == "/ask"
            assert root_attrs["app.component"] == "rag"
            assert root_attrs["rag.request.type"] == "ask"
            assert root_attrs["rag.model"] == "gpt-6-sol"
            assert root_attrs["rag.user_id"] == tenant["admin_id"]
            assert root_attrs["rag.session_id"]

            service_names = {attrs.get("service.name") for attrs, spans in resources if spans}
            assert service_names == {"rimini-rag-api"}
            assert all("service.namespace" not in attrs for attrs, spans in resources if spans)
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
            logger._traces = logger.TracerProvider()


def test_ingestion_exports_only_the_first_attempt(monkeypatch):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with _otlp_receiver() as (received, port):
        try:
            logger.configure_logging(
                settings=SimpleNamespace(
                    app_env="test",
                    otel_service_name="rimini-rag-api",
                    otel_exporter_otlp_protocol="http/protobuf",
                    otel_traces_sampler_arg="1.0",
                    otel_exporter_otlp_endpoint=f"http://127.0.0.1:{port}",
                    otel_exporter_otlp_headers="",
                )
            )

            @contextmanager
            def lease(*args):
                yield lambda: None

            monkeypatch.setattr(runner, "renewed_lease", lease)
            monkeypatch.setattr(runner, "_run_claimed_job", lambda *args: {})
            container = SimpleNamespace(
                queue=object(),
                settings=SimpleNamespace(job_lease_seconds=30),
            )
            container.for_workspace = lambda tenant_id: container
            correlation = {
                "traceparent": "00-11111111111111111111111111111111-2222222222222222-01"
            }
            base = dict(
                id=new_object_id(),
                document_id=new_object_id(),
                tenant_id=new_object_id(),
                stage="parse",
                status="running",
                correlation=correlation,
            )
            runner.run_job(container, Job(attempts=0, **base))
            runner.run_job(container, Job(attempts=1, **base))
            logger.shutdown_logging()

            spans = [
                span
                for _, resource_spans in _collect_spans(received)
                for span in resource_spans
            ]
            ingestion = [span for span in spans if span.name == "ingestion"]
            assert len(ingestion) == 1
            assert ingestion[0].trace_id.hex() == "11111111111111111111111111111111"
            assert _span_attrs(ingestion[0])["attempt"] == 0
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
            logger._traces = logger.TracerProvider()


def test_business_spans_do_not_create_orphan_traces():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with _otlp_receiver() as (received, port):
        try:
            logger.configure_logging(
                settings=SimpleNamespace(
                    app_env="test",
                    otel_service_name="rimini-rag-api",
                    otel_exporter_otlp_protocol="http/protobuf",
                    otel_traces_sampler_arg="1.0",
                    otel_exporter_otlp_endpoint=f"http://127.0.0.1:{port}",
                    otel_exporter_otlp_headers="",
                )
            )
            with logger.stage("retrieve_documents"):
                pass
            logger.shutdown_logging()

            spans = [
                span
                for _, resource_spans in _collect_spans(received)
                for span in resource_spans
            ]
            assert spans == []
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
            logger._traces = logger.TracerProvider()


def test_reprocess_route_is_traced_independent_of_identifier_validation():
    assert get_trace_request_type(reprocess) == "ingest"
    assert get_trace_request_type(ingest) == "ingest"


def test_ingestion_rejects_malformed_parent_without_creating_orphan_span(monkeypatch):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with _otlp_receiver() as (received, port):
        try:
            logger.configure_logging(
                settings=SimpleNamespace(
                    app_env="test",
                    otel_service_name="rimini-rag-api",
                    otel_exporter_otlp_protocol="http/protobuf",
                    otel_traces_sampler_arg="1.0",
                    otel_exporter_otlp_endpoint=f"http://127.0.0.1:{port}",
                    otel_exporter_otlp_headers="",
                )
            )

            @contextmanager
            def lease(*args):
                yield lambda: None

            monkeypatch.setattr(runner, "renewed_lease", lease)
            monkeypatch.setattr(runner, "_run_claimed_job", lambda *args: {})
            container = SimpleNamespace(
                queue=object(),
                settings=SimpleNamespace(job_lease_seconds=30),
            )
            container.for_workspace = lambda tenant_id: container
            runner.run_job(
                container,
                Job(
                    new_object_id(),
                    new_object_id(),
                    new_object_id(),
                    "parse",
                    "running",
                    0,
                    correlation={"traceparent": "malformed"},
                ),
            )
            logger.shutdown_logging()

            spans = [
                span
                for _, resource_spans in _collect_spans(received)
                for span in resource_spans
            ]
            assert spans == []
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
            logger._traces = logger.TracerProvider()
