"""Reusable observability helpers independent of API/business modules."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.propagate import extract

from app.shared.logger import log_context, stage

_TRACE_REQUEST_TYPE = "_trace_request_type"


def _trace_type_from_callable(endpoint: Any) -> str | None:
    current = endpoint
    while current is not None:
        value = getattr(current, _TRACE_REQUEST_TYPE, None)
        if value:
            return value
        current = getattr(current, "__wrapped__", None)
    return None


def trace_request(request_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a handler as root-traced and label it with a request type."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _TRACE_REQUEST_TYPE, request_type)
        return fn

    return decorator


def get_trace_request_type(endpoint: Any) -> str | None:
    """Return the tracing request type declared for an endpoint, if any."""
    return _trace_type_from_callable(endpoint)


def resolve_request_trace(scope: Mapping[str, Any], fallback_path: str) -> tuple[str, str] | None:
    """Resolve traced request metadata from the matched endpoint decorator."""
    endpoint = scope.get("endpoint")
    request_type = _trace_type_from_callable(endpoint)
    if not request_type:
        return None
    route_path = getattr(scope.get("route"), "path", None) or fallback_path
    return route_path, request_type


def _route_trace_candidates(routes: list[Any]):
    for route in routes:
        contexts = getattr(route, "effective_route_contexts", None)
        if callable(contexts):
            for context in contexts():
                request_type = _trace_type_from_callable(getattr(context, "endpoint", None))
                if not request_type:
                    continue
                methods = getattr(context, "methods", None)
                path_regex = getattr(context, "path_regex", None)
                path = getattr(context, "path", None)
                if methods and path_regex and path:
                    yield methods, path_regex, path, request_type
            continue
        request_type = _trace_type_from_callable(getattr(route, "endpoint", None))
        if not request_type:
            continue
        methods = getattr(route, "methods", None)
        path_regex = getattr(route, "path_regex", None)
        path = getattr(route, "path", None)
        if methods and path_regex and path:
            yield methods, path_regex, path, request_type


def resolve_request_trace_from_routes(
    routes: list[Any], method: str, path: str, fallback_path: str
) -> tuple[str, str] | None:
    """Resolve traced request metadata by matching method/path against decorated routes."""
    for methods, path_regex, route_path, request_type in _route_trace_candidates(routes):
        if method in methods and path_regex.match(path):
            return route_path, request_type
    return None


def is_quiet_path(path: str, *, quiet_paths: set[str], quiet_prefixes: tuple[str, ...]) -> bool:
    """Return True when request logging/tracing should stay quiet for this path."""
    return path in quiet_paths or path.startswith(quiet_prefixes)


def apply_request_span_attributes(
    detail: dict[str, Any],
    *,
    component: str | None,
    http_route: str,
    method: str,
    target: str,
    request_type: str | None,
    session_id: str,
    status_code: int,
    model: str | None = None,
    user_id: str | None = None,
) -> None:
    """Populate request-level span attributes in a shared, reusable place."""
    if component:
        detail["app.component"] = component
    detail["http.route"] = http_route
    detail["http.method"] = method
    detail["http.target"] = target
    if request_type:
        detail["rag.request.type"] = request_type
    detail["rag.session_id"] = session_id
    detail["http.response.status_code"] = status_code
    if model:
        detail["rag.model"] = model
    if user_id:
        detail["rag.user_id"] = user_id


def set_current_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Set attributes on the active span when present, otherwise no-op."""
    span = trace.get_current_span()
    if not span or not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None or not isinstance(value, (str, int, float, bool)):
            continue
        span.set_attribute(key, value)


def with_current_span_attributes(
    resolver: Callable[..., Mapping[str, Any]]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that resolves and applies current-span attributes for sync/async callables."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                set_current_span_attributes(resolver(*args, **kwargs))
                return await fn(*args, **kwargs)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            set_current_span_attributes(resolver(*args, **kwargs))
            return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


def trace_from_correlation(
    name: str,
    *,
    correlation: Callable[..., Mapping[str, Any] | None],
    fields: Callable[..., Mapping[str, Any]],
    should_trace: Callable[..., bool],
    pipeline: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for queue/background entrypoints that carry trace context in correlation data."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            parent = dict(correlation(*args, **kwargs) or {})
            parent_context = extract(parent)
            has_parent_trace = trace.get_current_span(parent_context).get_span_context().is_valid
            trace_this_call = should_trace(*args, has_parent_trace=has_parent_trace, **kwargs)
            with (
                log_context(**parent),
                stage(
                    name,
                    pipeline=pipeline,
                    parent=parent,
                    trace_span=trace_this_call,
                    allow_root=trace_this_call,
                    **dict(fields(*args, **kwargs)),
                ) as detail,
            ):
                result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    detail.update(result)
                return result

        return wrapper

    return decorator
