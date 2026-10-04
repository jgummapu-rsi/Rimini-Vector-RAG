from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.api.auth import get_container, get_principal
from app.shared.container import Container
from app.shared.domain.models import Principal
from app.shared.execution import ExecutionBudget, execution_scope
from app.shared.ports.request_gate import RequestCapacityError

log = logging.getLogger(__name__)


async def admit_request(
    request: Request,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
):
    gate = container.request_gate
    try:
        token = await run_in_threadpool(gate.acquire, principal.tenant_id, principal.user_id)
    except RequestCapacityError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "60"}) from exc
    except Exception as exc:
        log.error(
            "Request admission service unavailable",
            extra={"event": "admission_unavailable", "tenant_id": principal.tenant_id},
        )
        raise HTTPException(503, "Request admission service unavailable") from exc

    budget = ExecutionBudget(container.settings.request_timeout_seconds)

    async def renew():
        while True:
            await asyncio.sleep(20)
            try:
                await run_in_threadpool(gate.renew, principal.tenant_id, principal.user_id, token)
            except Exception:
                budget.cancel("Request admission lease could not be maintained")
                return

    renewal = asyncio.create_task(renew())
    try:
        if request.url.path in {"/query", "/ask", "/answer"}:
            with execution_scope(budget):
                yield
                budget.check()
        else:
            yield
    finally:
        renewal.cancel()
        await asyncio.gather(renewal, return_exceptions=True)
        try:
            await run_in_threadpool(gate.release, principal.tenant_id, principal.user_id, token)
        except Exception:
            log.warning(
                "Request admission slot release failed; lease will expire",
                extra={"event": "admission_release_failed", "tenant_id": principal.tenant_id},
            )
