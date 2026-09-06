"""FastAPI application factory with correlated request logging."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Optional

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.ingest_routes import router as ingest_router
from app.api.onboarding_routes import router as onboarding_router
from app.api.retrieval_routes import router as retrieval_router
from app.shared.container import Container, build_container
from app.shared.ids import new_object_id
from app.shared.observability import bind, configure_logging, reset

log = logging.getLogger("api")
_QUIET_PATHS = {"/healthz", "/metrics", "/openapi.json", "/docs", "/redoc", "/"}
# Static asset requests are noise in a correlated request log -- one page load is
# dozens of them, none of which say anything about the pipeline.
_QUIET_PREFIXES = ("/ui",)
_STATIC_DIR = Path(__file__).with_name("static") / "trace"
_ONBOARDING_STATIC_DIR = Path(__file__).with_name("static") / "onboarding"


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
                too_big = False           # unparseable header -> let the handler's cap decide
            if too_big:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": f"request body exceeds {max_mb} MB"},
                )
        return await call_next(request)

    @application.middleware("http")
    async def _correlate(request, call_next):
        request_id = new_object_id()
        token = bind(request_id=request_id, method=request.method, path=request.url.path)
        quiet = (request.url.path in _QUIET_PATHS
                 or request.url.path.startswith(_QUIET_PREFIXES))
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

    application.include_router(ingest_router)
    application.include_router(retrieval_router)
    application.include_router(onboarding_router)

    # Trace UI (app/api/static/trace/index.html). Mounted last so it can never
    # shadow an API route. Same-origin with the API, so its fetches need no CORS.
    if _STATIC_DIR.is_dir():
        trace_index = _STATIC_DIR / "index.html"

        # `/ui/trace` (and the redirect-normalized `/ui/trace/`) is documented
        # as this UI's canonical address, but StaticFiles(html=True) only ever
        # auto-serves `index.html` for a path that is a REAL directory on disk
        # (see starlette.staticfiles.StaticFiles.get_response) -- "trace" is
        # not a subdirectory of the static/trace dir, so that path 404ed with
        # no explicit route ahead of the mount below. Registered before the
        # `/ui` mount so it wins the match for these two exact paths; every
        # other /ui/* path still falls through to the mount unchanged. The app
        # itself is a single-page, hash-routed UI (`#/trace`, `#/ingest`), so
        # this alias serves the identical shell -- not a distinct server page.
        @application.get("/ui/trace", include_in_schema=False)
        @application.get("/ui/trace/", include_in_schema=False)
        def _trace_ui_alias() -> FileResponse:
            return FileResponse(trace_index)

        application.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    # Onboarding landing page (app/api/static/onboarding/index.html), served at
    # site root. Mounted absolutely last: a Mount("/") matches any unmatched
    # path, so if it were registered before the API routes it would shadow
    # every one of them -- same reasoning as the /ui mount above, just at root.
    if _ONBOARDING_STATIC_DIR.is_dir():
        application.mount("/", StaticFiles(directory=_ONBOARDING_STATIC_DIR, html=True),
                          name="onboarding")

    return application


app = create_app()
