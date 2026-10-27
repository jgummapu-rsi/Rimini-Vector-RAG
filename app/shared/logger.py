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
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_traces = TracerProvider()
_logs = None
_service_name = "rimini-rag"
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
def stage(name: str, *, logger=None, pipeline=False, parent=None, emit=True, **fields):
    """One completion event per real operation, including failures and elapsed time."""
    logger = logger or logging.getLogger("pipeline")
    owner = pipeline and _usage.get() is None
    usage_token = _usage.set(Usage()) if owner else None
    token = bind(stage=name, **fields)
    started = time.perf_counter()
    details = {}
    with _traces.get_tracer("rimini-rag").start_as_current_span(
        name, context=extract(parent) if parent else None
    ) as span:
        status = "ok"
        try:
            yield details
        except BaseException as exc:
            status = "error"
            details["error_type"] = type(exc).__name__
            span.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            details.update(duration_ms=(time.perf_counter() - started) * 1000, status=status)
            if owner:
                details.update(_usage.get().summary())
            for key, value in {**(_ctx.get() or {}), **details}.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
            if emit:
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
    """Log the actual packed evidence for each call, including retries and quote repair."""
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            logging.getLogger("app.retrieval.rag.query").info(
                "LLM evidence",
                extra={
                    "event": "llm_evidence",
                    "chunks": len(payload["evidence"]),
                    "evidence": payload["evidence"],
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
    """Compact summaries; evidence is rendered verbatim, never escaped or shortened.

    Full correlation stays on the record for OTLP filtering. Show the request/job
    on each summary and the trace on completion so console readers can follow a run.
    """

    @staticmethod
    def _value(value):
        if value is None:
            return "unknown"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, dict):
            return ", ".join(f"{k}: {TextFormatter._value(v)}" for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return ", ".join(TextFormatter._value(v) for v in value) or "none"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]
        component = getattr(record, "stage", record.name.rsplit(".", 1)[-1])
        parts = [f"{timestamp}Z", f"{record.levelname:<7}", component, record.getMessage()]
        if hasattr(record, "duration_ms"):
            parts.append(f"{record.duration_ms:,.3f} ms")
        for key in (
            "chunks",
            "vectors",
            "candidates",
            "model",
            "input_tokens",
            "output_tokens",
            "llm_calls",
            "cost_usd",
            "total_cost_usd",
        ):
            if hasattr(record, key):
                value = getattr(record, key)
                rendered = (
                    f"${value:.8f}"
                    if key.endswith("cost_usd") and value is not None
                    else self._value(value)
                )
                parts.append(f"{key.replace('_', ' ')}={rendered}")
        for key in ("job_id", "request_id"):
            if getattr(record, key, None):
                parts.append(f"{key.removesuffix('_id')}={getattr(record, key)}")
        lines = [" | ".join(parts)]
        hidden = _STD | {
            "stage",
            "event",
            "duration_ms",
            "service_name",
            "traceparent",
            "tracestate",
            "trace_id",
            "span_id",
            "request_id",
            "session_id",
            "tenant_id",
            "user_id",
            "client_ip",
            "job_id",
            "document_id",
            "generation_id",
            "role",
            "evidence",
            "chunks",
            "vectors",
            "candidates",
            "model",
            "input_tokens",
            "output_tokens",
            "llm_calls",
            "cost_usd",
            "total_cost_usd",
        }
        details = [
            f"{key.replace('_', ' ')}={self._value(value)}"
            for key, value in record.__dict__.items()
            if key not in hidden and not key.startswith("_") and value is not None
        ]
        if details:
            lines.append("  " + " | ".join(details))
        if getattr(record, "event", "") in {"pipeline_done", "request_end"}:
            ids = [
                f"{key}={getattr(record, key)}"
                for key in (
                    "trace_id",
                    "session_id",
                    "tenant_id",
                    "user_id",
                    "document_id",
                    "generation_id",
                )
                if getattr(record, key, None)
            ]
            if ids:
                lines.append("  " + " | ".join(ids))
        evidence = getattr(record, "evidence", None)
        if isinstance(evidence, list):
            lines.append("  BEGIN LLM EVIDENCE")
            for entry in evidence:
                if isinstance(entry, dict):
                    metadata = " | ".join(
                        f"{k.replace('_', ' ')}={self._value(v)}"
                        for k, v in entry.items()
                        if k != "text"
                    )
                    lines.append(f"  Source: {metadata}" if metadata else "  Source:")
                    lines.append(str(entry.get("text", "")))
                else:
                    lines.append(str(entry))
            lines.append("  END LLM EVIDENCE")
        if record.exc_info:
            lines.append(self.formatException(record.exc_info))
        return "\n".join(lines)


def configure_logging(
    level: int = logging.INFO, *, settings=None, service_name="rimini-rag"
) -> None:
    """Configure text stderr and optional batched Observer OTLP transport."""
    global _logs, _traces, _service_name
    shutdown_logging()
    _service_name = service_name
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "rimini",
            "deployment.environment.name": getattr(settings, "app_env", "local"),
        }
    )
    _traces = TracerProvider(resource=resource)
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
        transport.setFormatter(TextFormatter())
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


def shutdown_logging():
    global _logs
    if _logs is not None:
        _logs.shutdown()
        _logs = None
    _traces.shutdown()
