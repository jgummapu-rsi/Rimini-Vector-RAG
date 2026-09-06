"""LiteLLM gateway client (OpenAI-compatible): chat, embeddings, and vision
(there's no dedicated OCR model, so scanned-page transcription goes through
vision too -- see app.ingest.pipeline.prompts.STRICT_TRANSCRIBE_PROMPT).

Includes simple retry with exponential backoff for transient gateway errors.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Callable, Optional

import httpx

log = logging.getLogger("gateway")


class GatewayError(RuntimeError):
    """A gateway call failed. `status_code` is the HTTP status when the gateway
    actually answered, or None for a transport-level failure."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """Build the error with its `message` and optional HTTP `status_code`."""
        super().__init__(message)
        self.status_code = status_code


# Statuses worth trying again: the gateway is rate-limiting us, timed out, or is
# briefly unhealthy. Everything else in 4xx is a defect in the REQUEST -- a bad
# API key, an unknown model, a malformed payload -- and will fail identically
# however many times it is resent, so retrying only delays the error and
# triples the log noise.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


def _is_retryable(exc: Exception) -> bool:
    """True if `exc` is a transient failure worth retrying (see _RETRYABLE_STATUS)."""
    if isinstance(exc, httpx.TransportError):
        return True                       # connection reset / DNS / timeout
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
        config_provider: Optional[ConfigProvider] = None,
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

    def _resolve_config(self) -> tuple[str, str]:
        """Base URL + API key for the NEXT call. Re-read on every _post() call
        (not cached) -- a DB override (POST /onboarding/gateway-config,
        persisted in system_config) wins over the env-sourced constructor
        defaults so a live edit takes effect immediately, no restart. Falls
        back to the env default field-by-field if the override is blank."""
        if self._config_provider is not None:
            db_base_url, db_api_key = self._config_provider()
            if db_base_url or db_api_key:
                return (db_base_url.rstrip("/") if db_base_url else self._default_base_url,
                        db_api_key or self._default_api_key)
        return self._default_base_url, self._default_api_key

    def _headers(self, api_key: str) -> dict[str, str]:
        """Bearer-auth JSON headers for a request using `api_key`."""
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, payload: dict) -> dict:
        """POST `payload` to `path`, retrying transient failures with backoff."""
        base_url, api_key = self._resolve_config()
        url = f"{base_url}{path}"
        model = payload.get("model")
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=self._headers(api_key), json=payload)
                dur_ms = round((time.perf_counter() - t0) * 1000, 1)
                if resp.status_code >= 400:
                    raise GatewayError(f"{resp.status_code}: {resp.text[:300]}",
                                       status_code=resp.status_code)
                log.info("llm call", extra={"event": "llm_call", "path": path,
                         "model": model, "duration_ms": dur_ms, "attempt": attempt})
                return resp.json()
            except (httpx.TransportError, GatewayError) as e:
                last = e
                if not _is_retryable(e):
                    # Permanent: fail now with the gateway's own message intact,
                    # rather than burying it under "failed after N attempts".
                    log.error("llm call rejected", extra={
                        "event": "llm_rejected", "path": path, "model": model,
                        "status": getattr(e, "status_code", None), "error": str(e)[:300],
                    })
                    raise
                log.warning("llm retry", extra={"event": "llm_retry", "path": path,
                            "model": model, "attempt": attempt, "error": str(e)[:160]})
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        log.error("llm failed", extra={"event": "llm_failed", "path": path,
                  "model": model, "attempts": self.max_retries})
        raise GatewayError(f"gateway call failed after {self.max_retries} attempts: {last}",
                           status_code=getattr(last, "status_code", None))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one float32 embedding per input text."""
        data = self._post("/v1/embeddings", {"model": self.embedding_model, "input": texts})
        return [item["embedding"] for item in data["data"]]

    def chat(self, messages: list[dict], model: str, temperature: float = 0.0) -> str:
        """Chat completion (used for RAG answer generation)."""
        data = self._post("/v1/chat/completions", {
            "model": model, "messages": messages, "temperature": temperature,
        })
        return data["choices"][0]["message"]["content"] or ""

    def vision(self, image_bytes: bytes, prompt: str, mime: str = "image/png") -> str:
        """Transcribe an image via the vision LLM (this is our OCR path — the
        gateway has no dedicated OCR model). Returns the model's text output."""
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.vision_model,
            "temperature": 0,   # deterministic transcription, minimise invention
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        }
        data = self._post("/v1/chat/completions", payload)
        return data["choices"][0]["message"]["content"] or ""
