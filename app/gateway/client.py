"""LiteLLM gateway client (OpenAI-compatible): chat, embeddings, and vision
(there's no dedicated OCR model, so scanned-page transcription goes through
vision too -- see app.pipeline.prompts.STRICT_TRANSCRIBE_PROMPT).

Includes simple retry with exponential backoff for transient gateway errors.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import httpx

log = logging.getLogger("gateway")


class GatewayError(RuntimeError):
    pass


class LiteLLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        vision_model: str,
        embedding_model: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        model = payload.get("model")
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=self._headers(), json=payload)
                dur_ms = round((time.perf_counter() - t0) * 1000, 1)
                if resp.status_code >= 400:
                    raise GatewayError(f"{resp.status_code}: {resp.text[:300]}")
                log.info("llm call", extra={"event": "llm_call", "path": path,
                         "model": model, "duration_ms": dur_ms, "attempt": attempt})
                return resp.json()
            except (httpx.TransportError, GatewayError) as e:
                last = e
                log.warning("llm retry", extra={"event": "llm_retry", "path": path,
                            "model": model, "attempt": attempt, "error": str(e)[:160]})
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        log.error("llm failed", extra={"event": "llm_failed", "path": path,
                  "model": model, "attempts": self.max_retries})
        raise GatewayError(f"gateway call failed after {self.max_retries} attempts: {last}")

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
