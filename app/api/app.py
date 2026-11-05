"""FastAPI application factory with correlated request logging."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from app.api.body_limit import BodyLimitMiddleware
from app.api.ingest_routes import router as ingest_router
from app.api.onboarding_routes import router as onboarding_router
from app.api.retrieval_routes import router as retrieval_router
from app.shared.config import settings
from app.shared.container import Container, build_container
from app.shared.execution import RequestAborted
from app.shared.gateway.client import GatewayError
from app.shared.ids import new_object_id
from app.shared.logger import (
    bind,
    configure_logging,
    correlation,
    record_http_request,
    reset,
    shutdown_logging,
    stage,
)
from app.shared.observability import (
    apply_request_span_attributes,
    is_quiet_path,
    resolve_request_trace_from_routes,
)
from app.shared.ports.metadata_store import IngestionConflict

log = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).with_name("static") / "trace"
_ONBOARDING_STATIC_DIR = Path(__file__).with_name("static") / "onboarding"


def create_app(container: Container | None = None) -> FastAPI:
    """Build the app. If `container` is provided it is used as-is (tests inject a
    hermetic container); otherwise one is built from settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_settings = container.settings if container else settings
        configure_logging(
            settings=active_settings,
            service_name=active_settings.otel_api_service_name,
        )
        app.state.container = container if container is not None else build_container()
        try:
            yield
        finally:
            if container is None:
                app.state.container.close()
            shutdown_logging()

    application = FastAPI(title="RAG Ingestion Layer", version="0.1.0", lifespan=lifespan)

    @application.get("/readyz")
    def readiness():
        try:
            application.state.container.readiness_check()
        except Exception:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready"}

    @application.exception_handler(IngestionConflict)
    async def ingestion_conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(RequestAborted)
    async def request_aborted(request, exc):
        log.warning(
            "Request execution stopped",
            extra={"event": "request_aborted", "status": exc.status_code},
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @application.exception_handler(GatewayError)
    async def gateway_failed(request, exc):
        status_code = 504 if exc.status_code == 408 else 502
        log.error(
            "Gateway failure prevented request completion",
            extra={
                "event": "gateway_request_failed",
                "status": exc.status_code,
                "error": str(exc)[:300],
            },
        )
        return JSONResponse(
            status_code=status_code, content={"detail": "Gateway could not complete the request"}
        )

    @application.middleware("http")
    async def _limit_body_size(request, call_next):
        """Reject an oversized upload on its declared Content-Length, before the
        body is read.

        This has to live in middleware, not in the route: by the time
        `/ingest`'s handler runs, FastAPI has already parsed the multipart body
        and spooled it to a temp file, so a check inside the handler is too late
        to prevent the resource use. The handler still caps its own read (a
        client can omit or understate Content-Length) -- this is the cheap
        outer guard, that is the authoritative one.
        """
        declared = request.headers.get("content-length")
        if declared is not None:
            container_ = getattr(request.app.state, "container", None)
            max_mb = container_.settings.max_upload_mb if container_ else 0
            try:
                too_big = max_mb and int(declared) > max_mb * 1024 * 1024
            except ValueError:
                too_big = False
            if too_big:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": f"request body exceeds {max_mb} MB"},
                )
        return await call_next(request)

    @application.middleware("http")
    async def _correlate(request, call_next):
        app_container = getattr(request.app.state, "container", None)
        app_settings = app_container.settings if app_container is not None else settings
        request_id = request.headers.get("x-request-id") or new_object_id()
        session_id = request.headers.get("x-session-id") or new_object_id()
        token = bind(
            request_id=request_id,
            session_id=session_id,
            client_ip=request.client.host if request.client else "",
            method=request.method,
            path=request.url.path,
        )
        quiet = is_quiet_path(
            request.url.path,
            quiet_paths=set(app_settings.observability_quiet_paths),
            quiet_prefixes=tuple(app_settings.observability_quiet_prefixes),
        )
        t0 = perf_counter()
        status = 500
        traced = resolve_request_trace_from_routes(
            request.app.router.routes,
            request.method,
            request.url.path,
            request.url.path,
        )
        http_route = traced[0] if traced is not None else request.url.path
        request_type = traced[1] if traced is not None else None
        try:
            with stage(
                f"{request.method} {http_route}",
                parent=dict(request.headers),
                emit=False,
                trace_span=traced is not None,
                allow_root=traced is not None,
                span_kind=trace.SpanKind.SERVER,
            ) as detail:
                try:
                    response = await call_next(request)
                    status = response.status_code
                    response.headers["X-Request-ID"] = request_id
                    response.headers["X-Session-ID"] = session_id
                    return response
                except BaseException as exc:
                    candidate = getattr(exc, "status_code", None)
                    if isinstance(candidate, int):
                        status = candidate
                    raise
                finally:
                    principal = getattr(request.state, "principal", None)
                    user_id = principal.user_id if principal is not None else None
                    model = request.query_params.get("model") if request.query_params else None
                    apply_request_span_attributes(
                        detail,
                        component=app_settings.observability_component,
                        http_route=http_route,
                        method=request.method,
                        target=request.url.path,
                        request_type=request_type,
                        session_id=str(correlation().get("session_id") or session_id),
                        status_code=status,
                        model=model,
                        user_id=user_id,
                    )
        finally:
            principal = getattr(request.state, "principal", None)
            if principal is not None:
                bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
            route = getattr(request.scope.get("route"), "path", None) or request.url.path
            duration_s = perf_counter() - t0
            record_http_request(request.method, route, status, duration_s)
            if not quiet:
                log.info(
                    "request end",
                    extra={
                        "event": "request_end",
                        "status": status,
                        "duration_ms": duration_s * 1000,
                    },
                )
            reset(token)

    application.include_router(ingest_router)
    application.add_middleware(BodyLimitMiddleware)
    application.include_router(retrieval_router)
    application.include_router(onboarding_router)

    if _STATIC_DIR.is_dir():
        trace_index = _STATIC_DIR / "index.html"

        @application.get("/ui/trace", include_in_schema=False)
        @application.get("/ui/trace/", include_in_schema=False)
        def _trace_ui_alias() -> FileResponse:
            return FileResponse(trace_index)

        application.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    if _ONBOARDING_STATIC_DIR.is_dir():
        application.mount(
            "/", StaticFiles(directory=_ONBOARDING_STATIC_DIR, html=True), name="onboarding"
        )

    return application


app = create_app()
