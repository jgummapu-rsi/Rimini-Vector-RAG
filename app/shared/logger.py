"""Readable pipeline logs and Observer transport with structured correlation attributes."""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import math
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

_traces = TracerProvider()
_logs = None
_metrics = None
_http_requests = None
_http_duration = None
_service_name = "unknown-service"
_usage = contextvars.ContextVar("pipeline_usage", default=None)

_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}

_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("log_ctx", default=None)


def bind(**fields):
    """Merge fields into the current log context; returns a reset token."""
    merged = {**(_ctx.get() or {}), **{k: v for k, v in fields.items() if v is not None}}
    return _ctx.set(merged)


def correlation() -> dict:
    """Serializable correlation only; safe to carry through the durable job queue."""
    fields = {
        k: v
        for k, v in (_ctx.get() or {}).items()
        if k in {"request_id", "session_id", "tenant_id", "user_id", "client_ip"}
    }
    inject(fields)
    return fields


def record_http_request(method: str, route: str, status_code: int, duration_s: float) -> None:
    """Emit bounded HTTP RED metrics when an OTLP metrics exporter is configured."""
    if _http_requests is None or _http_duration is None:
        return
    attributes = {
        "http.request.method": method,
        "http.route": route,
        "http.response.status_code": status_code,
    }
    _http_requests.add(1, attributes)
    _http_duration.record(duration_s, attributes)


def gateway_headers() -> dict:
    fields = correlation()
    headers = {k: fields[k] for k in ("traceparent", "tracestate") if k in fields}
    for field in ("request_id", "session_id", "tenant_id"):
        if fields.get(field):
            headers["x-" + field.replace("_", "-")] = str(fields[field])
    if fields.get("session_id"):
        headers["x-litellm-session-id"] = str(fields["session_id"])
        headers["x-litellm-tags"] = f"{fields['session_id']},{fields.get('tenant_id', '')}"
    return headers


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.cost = 0.0
        self.unknown = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def summary(self):
        with self.lock:
            return {
                "llm_calls": self.calls,
                "known_cost_usd": self.cost,
                "total_cost_usd": None if self.unknown else self.cost,
                "cost_unknown_calls": self.unknown,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }


@contextmanager
def stage(
    name,
    *,
    logger=None,
    pipeline=False,
    parent=None,
    emit=None,
    trace_span=True,
    allow_root=False,
    span_kind=trace.SpanKind.INTERNAL,
    **fields,
):
    """Trace one operation; log pipeline boundaries and failures, not successful internals."""
    logger = logger or logging.getLogger("pipeline")
    owner = pipeline and _usage.get() is None
    usage_token = _usage.set(Usage()) if owner else None
    token = bind(stage=name, **fields)
    started = time.perf_counter()
    details = {}
    span_context = extract(parent) if parent else None
    parent_span = trace.get_current_span(span_context).get_span_context()
    create_span = trace_span and (allow_root or parent_span.is_valid)
    with (
        _traces.get_tracer(_service_name).start_as_current_span(
            name,
            context=span_context,
            kind=span_kind,
        )
        if create_span
        else _null_span()
    ) as span:
        status = "ok"
        try:
            yield details
        except BaseException as exc:
            status = "error"
            details["error_type"] = type(exc).__name__
            if create_span:
                span.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            details.update(duration_ms=(time.perf_counter() - started) * 1000, status=status)
            if owner:
                details.update(_usage.get().summary())
            if create_span:
                for key, value in {**(_ctx.get() or {}), **details}.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
            if status == "error" or (pipeline if emit is None else emit):
                logger.log(
                    logging.INFO if status == "ok" else logging.ERROR,
                    f"{name} {'done' if status == 'ok' else 'failed'}",
                    extra={
                        "event": "pipeline_done" if owner else "stage",
                        "stage": name,
                        **details,
                    },
                )
            reset(token)
            if usage_token is not None:
                _usage.reset(usage_token)


def measured(name, *, pipeline=False):
    """Time sync/async module entry points without changing their public signatures."""

    def decorate(fn):
        signature = inspect.signature(fn)

        def scope(args, kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            fields = {
                key: bound[key]
                for key in ("tenant_id", "user_id", "model")
                if bound.get(key) is not None
            }
            return stage(name, logger=logging.getLogger(fn.__module__), pipeline=pipeline, **fields)

        @wraps(fn)
        def sync(*args, **kwargs):
            with scope(args, kwargs) as detail:
                result = fn(*args, **kwargs)
                if hasattr(result, "chunk_ids"):
                    detail["chunks"] = len(result.chunk_ids)
                if hasattr(result, "answer_status"):
                    detail["answer_status"] = result.answer_status
                if name == "reranking":
                    detail["candidates"] = len(signature.bind(*args, **kwargs).arguments["hits"])
                    detail["chunks"] = len(result[0])
                return result

        @wraps(fn)
        async def asynchronous(*args, **kwargs):
            with scope(args, kwargs):
                return await fn(*args, **kwargs)

        return asynchronous if inspect.iscoroutinefunction(fn) else sync

    return decorate


def record_usage(data: dict, headers, model: str, duration_ms: float):
    usage = data.get("usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    cost = headers.get("x-litellm-response-cost")
    if cost is None:
        cost = usage.get("cost", data.get("response_cost"))
    try:
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0:
            cost = None
    except (TypeError, ValueError):
        cost = None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    totals = _usage.get()
    if totals is not None:
        with totals.lock:
            totals.calls += 1
            totals.cost += cost or 0
            totals.unknown += cost is None
            totals.input_tokens += input_tokens if type(input_tokens) is int else 0
            totals.output_tokens += output_tokens if type(output_tokens) is int else 0
    logging.getLogger("app.shared.gateway.client").info(
        "model call done",
        extra={
            "event": "llm_call",
            "model": model,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "cost_source": "gateway" if cost is not None else "unknown",
            "provider_response_id": data.get("id"),
        },
    )


def log_evidence(messages):
    """Log evidence volume without exporting retrieved document content."""
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            logging.getLogger("app.retrieval.rag.query").info(
                "LLM evidence prepared",
                extra={
                    "event": "llm_evidence_prepared",
                    "chunks": len(payload["evidence"]),
                },
            )


def configure_child_logging(send):
    """Relay native-parser events to the parent's single console/OTLP logger."""

    class Relay(logging.Handler):
        def emit(self, record):
            _ContextFilter().filter(record)
            send(
                {
                    "kind": "log",
                    "record": {
                        "name": record.name,
                        "levelno": record.levelno,
                        "levelname": record.levelname,
                        "msg": record.getMessage(),
                        "args": (),
                        "created": record.created,
                        **{
                            k: v
                            for k, v in record.__dict__.items()
                            if k not in _STD and not k.startswith("_")
                        },
                    },
                }
            )

    root = logging.getLogger()
    root.handlers[:] = [Relay()]
    root.setLevel(logging.INFO)


def reset(token) -> None:
    """Undo a prior `bind()` using the token it returned; no-op if already reset/stale."""
    try:
        _ctx.reset(token)
    except (ValueError, LookupError):
        pass


@contextmanager
def _null_span():
    yield None


@contextmanager
def log_context(**fields):
    """Bind `fields` into the log context for the duration of the `with` block, then reset."""
    token = bind(**fields)
    try:
        yield
    finally:
        reset(token)


class _ContextFilter(logging.Filter):
    """Copies the current log context's fields onto every LogRecord that doesn't already have them."""

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in (_ctx.get() or {}).items():
            if not hasattr(record, k):
                setattr(record, k, v)
        span = trace.get_current_span().get_span_context()
        record.service_name = _service_name
        if span.is_valid:
            record.trace_id = f"{span.trace_id:032x}"
            record.span_id = f"{span.span_id:016x}"
        return True


class TextFormatter(logging.Formatter):
    """Compact single-line logs for terminals; OTLP retains structured attributes."""

    @staticmethod
    def _value(value):
        if value is None:
            return "unknown"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    @classmethod
    def _field(cls, key, value) -> str:
        rendered = cls._value(value)
        if any(char.isspace() for char in rendered) or '"' in rendered:
            return f"{key}={json.dumps(rendered, ensure_ascii=False)}"
        return f"{key}={rendered}"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        stage = getattr(record, "stage", record.name.rsplit(".", 1)[-1])
        parts = [f"{timestamp}Z", f"{record.levelname:<7}", stage]
        if trace_id := getattr(record, "trace_id", None):
            parts.append(f"trace_id={trace_id}")
        if record.levelno >= logging.ERROR and (span_id := getattr(record, "span_id", None)):
            parts.append(f"span_id={span_id}")
        parts.append(record.getMessage())
        hidden = _STD | {
            "service_name",
            "traceparent",
            "tracestate",
            "trace_id",
            "span_id",
            "evidence",
        }
        fields = [
            self._field(key, value)
            for key, value in record.__dict__.items()
            if key not in hidden and not key.startswith("_") and value is not None
        ]
        if fields:
            parts.append(" ".join(fields))
        if record.exc_info:
            parts.append(self._field("exception", self.formatException(record.exc_info)))
        return " | ".join(parts)


def configure_logging(
    level: int = logging.INFO, *, settings=None, service_name=None
) -> None:
    """Configure JSON stderr and optional batched OTLP logs and traces."""
    global _logs, _metrics, _traces, _service_name, _http_requests, _http_duration
    shutdown_logging()
    resolved_service_name = (
        service_name
        or getattr(settings, "otel_service_name", None)
        or "unknown-service"
    )
    _service_name = resolved_service_name
    sample_arg = getattr(settings, "otel_traces_sampler_arg", "0.1")
    try:
        sample_ratio = float(sample_arg)
    except (TypeError, ValueError):
        sample_ratio = 0.1
    sample_ratio = min(1.0, max(0.0, sample_ratio))
    resource = Resource.create(
        {
            "service.name": resolved_service_name,
            "deployment.environment.name": getattr(settings, "app_env", "local"),
        }
    )
    _traces = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sample_ratio)),
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(TextFormatter())
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
    endpoint = getattr(settings, "otel_exporter_otlp_endpoint", "")
    if endpoint:
        protocol = getattr(settings, "otel_exporter_otlp_protocol", "http/protobuf")
        if protocol != "http/protobuf":
            raise ValueError("This application exports OTLP over HTTP/protobuf")
        headers = dict(
            item.strip().split("=", 1)
            for item in settings.otel_exporter_otlp_headers.split(",")
            if "=" in item
        )
        _logs = LoggerProvider(resource=resource)
        _logs.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(
                    endpoint=endpoint.rstrip("/") + "/v1/logs", headers=headers, timeout=5
                )
            )
        )
        transport = LoggingHandler(logger_provider=_logs)
        # OTLP already exports LogRecord fields as attributes. Keep the body
        # readable instead of duplicating those attributes as a JSON string.
        transport.setFormatter(logging.Formatter("%(message)s"))
        transport.addFilter(_ContextFilter())
        # Do not feed exporter failures back into the failing exporter.
        transport.addFilter(lambda record: record.name.startswith(("app.", "api", "pipeline")))
        root.addHandler(transport)
        _traces.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=endpoint.rstrip("/") + "/v1/traces", headers=headers, timeout=5
                )
            )
        )
        _metrics = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=endpoint.rstrip("/") + "/v1/metrics", headers=headers, timeout=5
                    ),
                    export_interval_millis=getattr(settings, "otel_metric_export_interval", 15000),
                )
            ],
        )
        meter = _metrics.get_meter(_service_name)
        _http_requests = meter.create_counter(
            "http.server.request.count",
            unit="{request}",
            description="Number of HTTP requests completed by the API.",
        )
        _http_duration = meter.create_histogram(
            "http.server.request.duration",
            unit="s",
            description="Duration of HTTP requests handled by the API.",
        )


def shutdown_logging():
    global _logs, _metrics, _http_requests, _http_duration
    if _logs is not None:
        _logs.shutdown()
        _logs = None
    if _metrics is not None:
        _metrics.shutdown()
        _metrics = None
    _http_requests = None
    _http_duration = None
    _traces.shutdown()
