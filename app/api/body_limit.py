from __future__ import annotations

from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        container = getattr(scope["app"].state, "container", None)
        if container is None:
            await self.app(scope, receive, send)
            return
        limit = container.settings.max_upload_mb * 1024 * 1024
        consumed = 0
        exceeded = False

        async def capped_receive():
            nonlocal consumed, exceeded
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    exceeded = True
                    raise ValueError("Request body exceeds upload limit")
            return message

        async def capped_send(message):
            if exceeded:
                return
            await send(message)

        try:
            await self.app(scope, capped_receive, capped_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await JSONResponse({"detail": "request body exceeds upload limit"}, status_code=413)(
                scope, receive, send
            )
