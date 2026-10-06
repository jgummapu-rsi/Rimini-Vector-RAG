"""LiteLLM gateway client (OpenAI-compatible): chat, embeddings, and vision
(there's no dedicated OCR model, so scanned-page transcription goes through
vision too -- see app.ingest.pipeline.prompts.STRICT_TRANSCRIBE_PROMPT).

Includes simple retry with exponential backoff for transient gateway errors.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from collections.abc import Callable

import httpx

from app.shared.domain.embedding import validate_vectors
from app.shared.execution import cancellable, check_execution, remaining_seconds

log = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    """A gateway call failed. `status_code` is the HTTP status when the gateway
    actually answered, or None for a transport-level failure."""

    def __init__(self, message: str, status_code: int | None = None):
        """Build the error with its `message` and optional HTTP `status_code`."""
        super().__init__(message)
        self.status_code = status_code


class VisionText(str):
    """Transcription with billed completion usage; remains string-compatible."""

    def __new__(cls, text: str, completion_tokens: int):
        result = super().__new__(cls, text)
        result.completion_tokens = completion_tokens
        return result


_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


def _is_retryable(exc: Exception) -> bool:
    """True if `exc` is a transient failure worth retrying (see _RETRYABLE_STATUS)."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, GatewayError):
        code = exc.status_code
        return code is not None and (code >= 500 or code in _RETRYABLE_STATUS)
    return False


ConfigProvider = Callable[[], tuple[str, str]]


class LiteLLMClient:
    """OpenAI-compatible LiteLLM HTTP client for chat, embeddings, and vision."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        vision_model: str,
        embedding_model: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
        config_provider: ConfigProvider | None = None,
    ):
        """Store connection defaults. `config_provider`, if given, is queried on
        every call for a live DB override (see `_resolve_config`); env-sourced
        `base_url`/`api_key` are the fallback whenever it returns nothing (or
        isn't supplied at all -- every test that constructs LiteLLMClient
        directly keeps working unchanged)."""
        self._default_base_url = base_url.rstrip("/")
        self._default_api_key = api_key
        self._config_provider = config_provider
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries
        if not math.isfinite(timeout) or timeout <= 0 or max_retries < 1:
            raise ValueError("Gateway timeout and retry count must be positive")

    def _resolve_config(self) -> tuple[str, str]:
        """Base URL + API key for the NEXT call. Re-read on every _post() call
        (not cached) -- a DB override (POST /onboarding/gateway-config,
        persisted in system_config) wins over the env-sourced constructor
        defaults so a live edit takes effect immediately, no restart. Falls
        back to the env default field-by-field if the override is blank."""
        if self._config_provider is not None:
            db_base_url, db_api_key = self._config_provider()
            if db_base_url or db_api_key:
                if not db_base_url or not db_api_key:
                    raise GatewayError("Gateway override must contain both URL and key")
                return db_base_url.rstrip("/"), db_api_key
        return self._default_base_url, self._default_api_key

    def _headers(self, api_key: str) -> dict[str, str]:
        """Bearer-auth JSON headers for a request using `api_key`."""
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, payload: dict) -> dict:
        """POST `payload` to `path`, retrying transient failures with backoff."""
        base_url, api_key = self._resolve_config()
        url = f"{base_url}{path}"
        model = payload.get("model")
        last: Exception | None = None
        deadline = time.monotonic() + remaining_seconds(self.timeout)
        for attempt in range(self.max_retries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GatewayError("Gateway request deadline exhausted", status_code=408) from last
            t0 = time.perf_counter()
            try:
                check_execution()
                resp = asyncio.run(cancellable(self._request(url, api_key, payload, remaining)))
                if time.monotonic() > deadline:
                    raise GatewayError("Gateway request deadline exhausted", status_code=408)
                dur_ms = round((time.perf_counter() - t0) * 1000, 1)
                if resp.status_code >= 400:
                    raise GatewayError(
                        f"{resp.status_code}: {resp.text[:300]}", status_code=resp.status_code
                    )
                try:
                    data = resp.json()
                except (TypeError, ValueError) as exc:
                    raise GatewayError("gateway returned invalid JSON", status_code=502) from exc
                if not isinstance(data, dict):
                    raise GatewayError(
                        "gateway returned a non-object JSON response", status_code=502
                    )
                log.debug(
                    "llm call",
                    extra={
                        "event": "llm_call",
                        "path": path,
                        "model": model,
                        "duration_ms": dur_ms,
                        "attempt": attempt,
                    },
                )
                return data
            except TimeoutError as exc:
                check_execution()
                raise GatewayError("Gateway request deadline exhausted", status_code=408) from exc
            except (httpx.TransportError, GatewayError) as e:
                last = e
                if not _is_retryable(e):
                    log.error(
                        "llm call rejected",
                        extra={
                            "event": "llm_rejected",
                            "path": path,
                            "model": model,
                            "status": getattr(e, "status_code", None),
                            "error": str(e)[:300],
                        },
                    )
                    raise
                log.warning(
                    "llm retry",
                    extra={
                        "event": "llm_retry",
                        "path": path,
                        "model": model,
                        "attempt": attempt,
                        "error": str(e)[:160],
                    },
                )
                if attempt < self.max_retries - 1:
                    if deadline - time.monotonic() <= 2**attempt:
                        raise GatewayError(
                            "Gateway retry would exceed request deadline", status_code=408
                        ) from e
                    time.sleep(2**attempt)
        log.error(
            "llm failed",
            extra={
                "event": "llm_failed",
                "path": path,
                "model": model,
                "attempts": self.max_retries,
            },
        )
        raise GatewayError(
            f"gateway call failed after {self.max_retries} attempts: {last}",
            status_code=getattr(last, "status_code", None),
        )

    async def _request(self, url: str, api_key: str, payload: dict, timeout: float):
        limit = 32 * 1024 * 1024 if url.endswith("/embeddings") else 2 * 1024 * 1024
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, headers=self._headers(api_key), json=payload
                ) as response:
                    body = bytearray()
                    async for block in response.aiter_bytes():
                        body.extend(block)
                        if len(body) > limit:
                            raise GatewayError("Gateway response exceeds the response-byte limit")
                    return httpx.Response(
                        response.status_code, headers=response.headers, content=bytes(body)
                    )

    def embed(
        self, texts: list[str], *, dimensions: int | None = None, model: str | None = None
    ) -> list[list[float]]:

        if not texts:
            return []
        payload = {"model": model or self.embedding_model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        data = self._post("/v1/embeddings", payload)
        try:
            items = data["data"]
            if not isinstance(items, list) or len(items) != len(texts):
                raise ValueError("Embedding response cardinality mismatch")
            indices = [item["index"] for item in items]
            if any(type(index) is not int for index in indices) or sorted(indices) != list(
                range(len(texts))
            ):
                raise ValueError("Embedding response indices must cover every input exactly once")
            vectors = [item["embedding"] for item in sorted(items, key=lambda item: item["index"])]
            return validate_vectors(vectors, len(texts), dimensions or len(vectors[0]))
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise GatewayError(f"Invalid embedding response: {exc}") from exc

    def chat(self, messages: list[dict], model: str, temperature: float = 0.0) -> str:
        """Chat completion (used for RAG answer generation)."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": 4096,
        }
        if model.startswith("gpt-5"):
            payload["reasoning_effort"] = "low"
        if model.startswith(("gpt-5.5", "gpt-5.6", "gpt-6")):
            payload.pop("temperature", None)
        data = self._post("/v1/chat/completions", payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
            refusal = message.get("refusal")
            finish_reason = choice.get("finish_reason")
            if refusal:
                raise GatewayError(f"Chat response was refused: {str(refusal)[:160]}")
            if finish_reason not in (None, "stop"):
                raise GatewayError(f"Chat response incomplete: finish_reason={finish_reason}")
            if not isinstance(content, str) or not content.strip():
                raise GatewayError("Chat response must contain nonempty text")
        except (KeyError, TypeError, IndexError) as exc:
            raise GatewayError("Chat response has an invalid schema") from exc
        return content

    def vision(self, image_bytes: bytes, prompt: str, mime: str = "image/png") -> str:
        """Transcribe an image via the vision LLM (this is our OCR path — the
        gateway has no dedicated OCR model). Returns the model's text output."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.vision_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
        }
        if self.vision_model.startswith(("gpt-5.5", "gpt-5.6", "gpt-6")):
            payload.pop("temperature", None)
        data = self._post("/v1/chat/completions", payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayError("vision response has an invalid completion schema") from exc
        refusal = message.get("refusal")
        if refusal:
            raise GatewayError(f"vision response was refused: {str(refusal)[:160]}")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in (None, "stop"):
            raise GatewayError(f"vision response incomplete: finish_reason={finish_reason}")
        if not isinstance(content, str):
            raise GatewayError("vision response content is not text")
        return content

    async def vision_bounded(
        self, image_bytes: bytes, prompt: str, mime: str, *, timeout_seconds: float, max_tokens: int
    ) -> str:
        if timeout_seconds <= 0 or max_tokens <= 0:
            raise GatewayError("Vision request budget exhausted")
        base_url, api_key = self._resolve_config()
        payload = {
            "model": self.vision_model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
                            },
                        },
                    ],
                }
            ],
        }
        if self.vision_model.startswith(("gpt-5.5", "gpt-5.6", "gpt-6")):
            payload.pop("temperature", None)
        if self.vision_model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    headers=self._headers(api_key),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        raise GatewayError(
                            "Vision request rejected", status_code=response.status_code
                        )
                    body = bytearray()
                    async for block in response.aiter_bytes():
                        body.extend(block)
                        if len(body) > 2 * 1024 * 1024:
                            raise GatewayError("Vision response exceeds the response-byte limit")
        try:
            data = json.loads(body)
            choice = data["choices"][0]
            message = choice["message"]
            text = message["content"]
            if (
                choice.get("finish_reason") != "stop"
                or message.get("refusal")
                or not isinstance(text, str)
            ):
                raise GatewayError("Vision response is incomplete or refused")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GatewayError("Vision response has an invalid schema") from exc
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        # Missing usage must not bypass the spending guard. Completion usage
        # includes reasoning tokens when the provider reports them in its total.
        if type(completion_tokens) is not int or completion_tokens < 0:
            completion_tokens = max_tokens
        if completion_tokens > max_tokens:
            raise GatewayError("Vision completion usage exceeds its requested token limit")
        return VisionText(text, completion_tokens)
