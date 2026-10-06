"""Server-side model discovery and the supported embedding recipes."""

from __future__ import annotations

import time
from threading import Lock

import httpx

from app.shared.gateway.client import GatewayError

LOCAL_EMBEDDINGS = {
    "local:minilm": {
        "id": "local:minilm",
        "label": "MiniLM L6 v2 — local CPU",
        "provider": "local",
        "repo": "Xenova/all-MiniLM-L6-v2",
        "dimensions": 384,
        "pooling": "mean",
        "max_tokens": 256,
        "revision": "751bff37182d3f1213fa05d7196b954e230abad9",
    },
    "local:bge-small": {
        "id": "local:bge-small",
        "label": "BGE Small English v1.5 — local CPU",
        "provider": "local",
        "repo": "Xenova/bge-small-en-v1.5",
        "dimensions": 384,
        "pooling": "cls",
        "max_tokens": 512,
        "revision": "ea104dacec62c0de699686887e3f920caeb4f3e3",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    },
}
GATEWAY_EMBEDDINGS = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}


class ModelCatalog:
    def __init__(self, gateway):
        self.gateway = gateway
        self._lock = Lock()
        self._cached = None
        self._expires = 0.0
        self._identity = None

    def get(self) -> dict:
        identity = self.gateway._resolve_config()
        with self._lock:
            if (
                self._cached is not None
                and identity == self._identity
                and time.monotonic() < self._expires
            ):
                return self._cached
            base, key = identity
            try:
                response = httpx.get(
                    base.rstrip("/") + "/model/info",
                    headers={"Authorization": "Bearer " + key},
                    timeout=15,
                )
                response.raise_for_status()
                data = response.json()["data"]
                if not isinstance(data, list):
                    raise ValueError("Invalid model catalog")
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                raise GatewayError("The gateway model list is unavailable. Please retry.") from exc
            chat, embeddings = {}, {}
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("model_name")
                upstream = (entry.get("litellm_params") or {}).get("model", "")
                mode = (entry.get("model_info") or {}).get("mode")
                if not isinstance(name, str) or not name or not isinstance(upstream, str):
                    continue
                if "claude" in (name + " " + upstream).lower() or "anthropic/" in upstream.lower():
                    continue
                # This application uses chat/completions, not Responses-only models.
                if mode == "chat" and "/responses/" not in upstream:
                    chat[name] = {"id": name, "label": name}
                if mode == "embedding" and name in GATEWAY_EMBEDDINGS:
                    embeddings[name] = {
                        "id": name,
                        "label": name + " — LiteLLM",
                        "provider": "gateway",
                        "dimensions": 1536,
                    }
            result = {
                "chat_models": sorted(chat.values(), key=lambda item: item["id"]),
                "embedding_models": [
                    *sorted(embeddings.values(), key=lambda item: item["id"]),
                    *[
                        {k: value[k] for k in ("id", "label", "provider", "dimensions")}
                        for value in LOCAL_EMBEDDINGS.values()
                    ],
                ],
            }
            self._identity, self._cached, self._expires = identity, result, time.monotonic() + 60
            return result
