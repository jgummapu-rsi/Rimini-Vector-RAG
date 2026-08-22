"""
title: Rimini Street Knowledge Assistant
author: rag-ingestion
version: 1.0.0
description: Chat-native front end for the Rimini Street RAG pipeline. Attach a document to ingest it -- watch it flow live through parse, route, extract, chunk, metadata, embed, binarize and index -- or just ask a question to see the retrieval pipeline (decompose, hybrid retrieve, rerank, generate) plus cited sources.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx
from pydantic import BaseModel, Field

# Human-friendly labels for the ingestion pipeline's real stage names.
# Keep in sync with app.domain.models.JobStage.
_STAGE_LABELS = {
    "parse": "Parsing document",
    "route": "Routing by content type",
    "extract": "Extracting structure",
    "chunk": "Chunking",
    "metadata": "Extracting metadata",
    "embed": "Embedding chunks",
    "binarize": "Preparing vectors",
    "upsert": "Indexing into knowledge base",
    "done": "Indexed",
}
_TERMINAL_JOB_STATUSES = {"done", "failed", "dead"}
_EMPTY_TURN_TEXT = {"", "ok", "okay", "thanks"}


class Pipe:
    class Valves(BaseModel):
        RAG_BASE_URL: str = Field(
            default="http://localhost:8000",
            description="Base URL of the RAG ingestion/query FastAPI service.",
        )
        RAG_API_TOKEN: str = Field(
            default="",
            description="Bearer token for the RAG backend, from `python -m scripts.seed`.",
        )
        OPENWEBUI_BASE_URL: str = Field(
            default="http://localhost:8080",
            description="Open WebUI's own base URL, used to download chat attachment bytes.",
        )
        OPENWEBUI_API_KEY: str = Field(
            default="",
            description="Open WebUI API key (Settings > Account > API Keys), needed to "
                        "download attachment content before forwarding it to /ingest.",
        )
        TOP_K: int = Field(default=5, description="Chunks to retrieve per query.")
        POLL_INTERVAL_S: float = Field(default=1.5, description="Seconds between job-status polls.")
        POLL_TIMEOUT_S: float = Field(default=180.0, description="Give up waiting on a job after this long.")

    def __init__(self):
        self.valves = self.Valves()

    # ---------------------------------------------------------------- auth --

    def _rag_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.valves.RAG_API_TOKEN}"}

    # ---------------------------------------------------------- attachments --

    async def _read_attachment_bytes(self, file_id: str) -> tuple[bytes, str, str]:
        """Fetch original bytes for an Open WebUI chat attachment via Open
        WebUI's own public REST API -- deliberately not reaching into internal
        storage classes, which vary release to release."""
        if not self.valves.OPENWEBUI_API_KEY:
            raise RuntimeError(
                "OPENWEBUI_API_KEY valve is not set -- required to download chat "
                "attachments before forwarding them to /ingest."
            )
        headers = {"Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}"}
        async with httpx.AsyncClient(base_url=self.valves.OPENWEBUI_BASE_URL, timeout=30) as client:
            meta = await client.get(f"/api/v1/files/{file_id}", headers=headers)
            meta.raise_for_status()
            info = meta.json()
            content = await client.get(f"/api/v1/files/{file_id}/content", headers=headers)
            content.raise_for_status()
        filename = info.get("filename") or (info.get("meta") or {}).get("name") or file_id
        content_type = (info.get("meta") or {}).get("content_type", "application/octet-stream")
        return content.content, filename, content_type

    def _current_turn_attachments(self, body: dict, files: Optional[list]) -> list[dict]:
        """Only the attachments on the CURRENT turn. Open WebUI's global
        `__files__` kwarg accumulates every file from the whole conversation,
        which would otherwise re-trigger ingestion status noise on every
        follow-up question. The backend's own sha256 dedup (`/ingest`) makes
        any accidental re-submission a cheap no-op regardless, so this is an
        efficiency/UX choice, not a correctness requirement."""
        messages = body.get("messages") or []
        if messages:
            turn_files = messages[-1].get("files")
            if turn_files:
                return turn_files
        return files or []

    @staticmethod
    def _file_id(attachment: dict) -> Optional[str]:
        return attachment.get("id") or (attachment.get("file") or {}).get("id")

    # ------------------------------------------------------------- ingest --

    async def _ingest_one(self, emitter, attachment: dict) -> None:
        file_id = self._file_id(attachment)
        if not file_id:
            return
        try:
            await emitter({"type": "status", "data": {"description": "Reading attachment…", "done": False}})
            data, filename, content_type = await self._read_attachment_bytes(file_id)

            await emitter({"type": "status", "data": {"description": f"Uploading {filename}…", "done": False}})
            async with httpx.AsyncClient(base_url=self.valves.RAG_BASE_URL, timeout=60) as client:
                resp = await client.post(
                    "/ingest", headers=self._rag_headers(),
                    files={"file": (filename, data, content_type)},
                )
            if resp.status_code >= 400:
                await emitter({"type": "status", "data": {
                    "description": f"{filename}: ingest failed -- {resp.text}", "done": True}})
                return

            payload = resp.json()
            if payload.get("deduplicated"):
                await emitter({"type": "status", "data": {
                    "description": f"{filename}: identical content already indexed", "done": True}})
                return

            await self._track_job(emitter, payload["job_id"], filename)
        except Exception as exc:  # never let one bad attachment kill the turn
            await emitter({"type": "status", "data": {
                "description": f"Attachment failed: {exc}", "done": True}})

    async def _track_job(self, emitter, job_id: str, filename: str) -> None:
        deadline = time.monotonic() + self.valves.POLL_TIMEOUT_S
        seen_stage = None
        async with httpx.AsyncClient(base_url=self.valves.RAG_BASE_URL, timeout=30) as client:
            while time.monotonic() < deadline:
                r = await client.get(f"/jobs/{job_id}", headers=self._rag_headers())
                r.raise_for_status()
                job = r.json()
                stage, job_status = job["stage"], job["status"]

                if stage != seen_stage:
                    seen_stage = stage
                    label = _STAGE_LABELS.get(stage, stage)
                    await emitter({"type": "status", "data": {
                        "description": f"{filename}: {label}…", "done": False}})

                if job_status in _TERMINAL_JOB_STATUSES:
                    if job_status == "done":
                        await emitter({"type": "status", "data": {
                            "description": f"{filename}: indexed and ready", "done": True}})
                    else:
                        await emitter({"type": "status", "data": {
                            "description": f"{filename}: ingestion {job_status} -- "
                                           f"{job.get('error') or 'unknown error'}",
                            "done": True}})
                    return
                await asyncio.sleep(self.valves.POLL_INTERVAL_S)

        await emitter({"type": "status", "data": {
            "description": f"{filename}: still processing after "
                           f"{int(self.valves.POLL_TIMEOUT_S)}s -- check /jobs/{job_id} manually",
            "done": True}})

    # -------------------------------------------------------------- query --

    async def _answer_question(self, emitter, question: str) -> str:
        async with httpx.AsyncClient(base_url=self.valves.RAG_BASE_URL, timeout=60) as client:
            resp = await client.post(
                "/query", headers=self._rag_headers(),
                json={"question": question, "top_k": self.valves.TOP_K},
            )
        if resp.status_code >= 400:
            await emitter({"type": "status", "data": {
                "description": f"Query failed -- {resp.text}", "done": True}})
            return f"Sorry, the knowledge base returned an error: {resp.text}"

        result = resp.json()
        for step in result.get("trace", []):
            await emitter({"type": "status", "data": {"description": step["detail"], "done": False}})

        answer = result.get("answer", "")
        if result.get("grounded", True):
            for c in result.get("citations", []):
                label = c.get("filename") or c.get("document_id") or "source"
                if c.get("location"):
                    label = f"{label} ({c['location']})"
                await emitter({
                    "type": "citation",
                    "data": {
                        "document": [c.get("snippet", "")],
                        "metadata": [{"source": label, "score": c.get("score")}],
                        "source": {"name": label},
                    },
                })
        else:
            # Nothing relevant was found in the knowledge base -- the answer
            # is the model's own general knowledge, not sourced from any
            # document. Never attach citations to it (they'd falsely imply
            # the answer came from your documents); say so plainly instead.
            answer = f"*(Not found in your documents -- general knowledge answer)*\n\n{answer}"

        await emitter({"type": "status", "data": {"description": "Answer ready", "done": True}})
        return answer

    # ----------------------------------------------------------- entrypoint --

    async def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        __files__: Optional[list] = None,
    ):
        if __event_emitter__ is None:
            async def __event_emitter__(_event):
                return None

        if not self.valves.RAG_API_TOKEN:
            yield ("Rimini Street RAG Pipe is not configured -- set RAG_API_TOKEN "
                   "in its Valves (Admin Panel > Functions).")
            return

        messages = body.get("messages") or [{}]
        question = (messages[-1].get("content") or "").strip()
        attachments = self._current_turn_attachments(body, __files__)

        if attachments:
            for attachment in attachments:
                await self._ingest_one(__event_emitter__, attachment)
            if question.lower() in _EMPTY_TURN_TEXT:
                yield "Document received -- ask me anything about it."
                return

        yield await self._answer_question(__event_emitter__, question)
