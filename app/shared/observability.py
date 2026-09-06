"""Structured JSON logging with an auto-attached correlation context.

Every log line is one JSON object. A per-task/thread context (request_id,
tenant_id, user_id, job_id, document_id, stage, ...) is bound once at the edges
and then **auto-attached to every log line** within that scope — so you can grep
a single id and see the whole lifecycle: request -> ingest -> job -> stages ->
LLM calls -> done.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextlib import contextmanager

# standard LogRecord attributes we don't duplicate into the JSON body
_STD = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}

_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("log_ctx", default={})


def bind(**fields):
    """Merge fields into the current log context; returns a reset token."""
    merged = {**_ctx.get(), **{k: v for k, v in fields.items() if v is not None}}
    return _ctx.set(merged)


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
        for k, v in _ctx.get().items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


class JsonFormatter(logging.Formatter):
    """Renders each LogRecord as one JSON object, including any bound context fields."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Replace the root logger's handlers with a single JSON stderr handler at `level`."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
