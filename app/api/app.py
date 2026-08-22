"""FastAPI application factory with correlated request logging."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Optional

from fastapi import FastAPI

from app.api.routes import router
from app.container import Container, build_container
from app.ids import new_object_id
from app.observability import bind, configure_logging, reset

log = logging.getLogger("api")
_QUIET_PATHS = {"/healthz", "/metrics", "/openapi.json", "/docs", "/redoc"}


def create_app(container: Optional[Container] = None) -> FastAPI:
    """Build the app. If `container` is provided it is used as-is (tests inject a
    hermetic container); otherwise one is built from settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.container = container if container is not None else build_container()
        yield

    application = FastAPI(title="RAG Ingestion Layer", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def _correlate(request, call_next):
        request_id = new_object_id()
        token = bind(request_id=request_id, method=request.method, path=request.url.path)
        quiet = request.url.path in _QUIET_PATHS
        t0 = perf_counter()
        if not quiet:
            log.info("request start", extra={"event": "request_start"})
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            if not quiet:
                log.info("request end", extra={
                    "event": "request_end", "status": status,
                    "duration_ms": round((perf_counter() - t0) * 1000, 1),
                })
            reset(token)

    application.include_router(router)
    return application


app = create_app()
