from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import multiprocessing
import os
import resource
import select
import struct
import time
from collections.abc import Callable
from dataclasses import asdict
from types import SimpleNamespace

from app.ingest.pipeline.docling_config import options_dict, parser_profile, uses_docling
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.layout import detect_layout, layout_profile
from app.ingest.pipeline.loaders import extract_document
from app.ingest.pipeline.safety import UnsafeContentError, check_archive_expansion

log = logging.getLogger(__name__)
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
PARSER_REVISION = "docling-structured-elements-v6"


def _transfer(connection, data: bytes | None, size: int, deadline: float | None) -> bytes:
    fd = connection.fileno()
    os.set_blocking(fd, False)
    result = bytearray()
    offset = 0
    while offset < size:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Native parser IPC exceeded its deadline")
        readable, writable, _ = select.select(
            [fd] if data is None else [], [fd] if data is not None else [], [], remaining
        )
        if not readable and not writable:
            raise TimeoutError("Native parser IPC exceeded its deadline")
        try:
            if data is None:
                block = os.read(fd, min(size - offset, 65536))
                if not block:
                    raise EOFError("Native parser closed its IPC channel")
                result.extend(block)
                offset += len(block)
            else:
                offset += os.write(fd, memoryview(data)[offset : offset + 65536])
        except BlockingIOError:
            continue
    return bytes(result)


def _send(connection, message: dict, deadline: float | None = None) -> None:
    encoded = json.dumps(message, ensure_ascii=False).encode()
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise UnsafeContentError("Native extraction exceeded the IPC evidence limit")
    framed = struct.pack("!I", len(encoded)) + encoded
    _transfer(connection, framed, len(framed), deadline)


def _receive(connection, deadline: float | None = None) -> dict:
    length = struct.unpack("!I", _transfer(connection, None, 4, deadline))[0]
    if length > MAX_MESSAGE_BYTES:
        raise UnsafeContentError("Native parser IPC exceeded the message limit")
    message = json.loads(_transfer(connection, None, length, deadline))
    if not isinstance(message, dict):
        raise UnsafeContentError("Native parser IPC must contain an object")
    return message


class _VisionProxy:
    def __init__(self, connection):
        self.connection = connection

    def vision(self, image_bytes: bytes, prompt: str, mime: str = "image/png") -> str:
        _send(
            self.connection,
            {
                "kind": "vision",
                "image": base64.b64encode(image_bytes).decode(),
                "prompt": prompt,
                "mime": mime,
            },
        )
        response = _receive(self.connection)
        if response["kind"] != "vision_result":
            raise UnsafeContentError("Invalid parent extraction response")
        return response["text"]

    def layout(self, image_bytes: bytes) -> list[dict]:
        _send(self.connection, {"kind": "layout", "image": base64.b64encode(image_bytes).decode()})
        response = _receive(self.connection)
        if response["kind"] != "layout_result":
            raise UnsafeContentError("Invalid parent layout response")
        return response["regions"]


async def _vision_with_lease(
    gateway, image, prompt, mime, timeout_seconds, max_tokens, check_lease
):
    async def monitor():
        while True:
            if check_lease is not None:
                check_lease()
            await asyncio.sleep(0.1)

    request = asyncio.create_task(
        gateway.vision_bounded(
            image, prompt, mime, timeout_seconds=timeout_seconds, max_tokens=max_tokens
        )
    )
    watcher = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait({request, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher
        return await request
    finally:
        request.cancel()
        watcher.cancel()
        await asyncio.gather(request, watcher, return_exceptions=True)


def _native_child(connection, filename: str, data: bytes, config: dict) -> None:
    try:
        cfg = SimpleNamespace(**config)
        threads = cfg.docling_num_threads if uses_docling(filename, cfg) else 1
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(threads)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        ceiling = cfg.parse_memory_mb * 1024 * 1024
        # RLIMIT_AS cannot reclaim memory inherited from the forkserver. Reject
        # a budget already below resident usage before limiting error reporting.
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > ceiling:
            raise UnsafeContentError("Parser memory budget is below its resident runtime")
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        # RLIMIT_CPU sums CPU time across threads, unlike the wall-clock budget.
        cpu_seconds = max(1, math.ceil(cfg.parse_timeout_seconds * threads) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        check_archive_expansion(data, filename, cfg)
        elements, summary = extract_document(filename, data, _VisionProxy(connection), cfg)
        if sum(len(element.text.encode()) for element in elements) > cfg.max_extracted_bytes:
            raise UnsafeContentError("Extracted evidence exceeds the content limit")
        _send(
            connection,
            {"kind": "result", "elements": [asdict(e) for e in elements], "summary": summary},
        )
    except BaseException as exc:
        try:
            _send(connection, {"kind": "error", "error_type": type(exc).__name__})
        except (OSError, ValueError):
            pass
    finally:
        connection.close()


def extract_bounded(
    filename: str,
    data: bytes,
    gateway,
    cfg,
    load_artifact: Callable[[str], dict | None],
    save_artifact: Callable[[str, dict], None],
    check_lease: Callable[[], None] | None = None,
) -> tuple[list[Element], dict]:
    if os.name != "posix":
        raise RuntimeError("Bounded parsing requires the supported Linux runtime")
    parsing = parser_profile(filename, cfg)
    profile = {"enabled": False} if uses_docling(filename, cfg) else layout_profile(cfg)
    result_key = hashlib.sha256(
        json.dumps(
            {
                "revision": PARSER_REVISION,
                "filename": filename,
                "content": hashlib.sha256(data).hexdigest(),
                "model": getattr(gateway, "vision_model", ""),
                "layout": profile,
                "parsing": parsing,
                "limits": {
                    name: getattr(cfg, name)
                    for name in (
                        "max_pdf_pages",
                        "max_image_pixels",
                        "max_zip_entries",
                        "max_zip_expanded_bytes",
                        "max_zip_expansion_ratio",
                        "max_extracted_bytes",
                        "max_vision_calls",
                        "max_vision_tokens",
                        "vision_max_tokens",
                        "max_table_rows",
                        "max_workbook_cells",
                        "max_workbook_sheets",
                        "parse_memory_mb",
                        "parse_timeout_seconds",
                    )
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    prior = load_artifact(result_key)
    if prior is not None:
        summary = dict(prior["summary"], extraction_reused=True, vision_calls=0)
        return [Element(**element) for element in prior["elements"]], summary
    multiprocessing.set_forkserver_preload(["app.ingest.pipeline.loaders"])
    context = multiprocessing.get_context("forkserver")
    parent, child = context.Pipe()
    native_config = {
        name: getattr(cfg, name)
        for name in (
            "parse_memory_mb",
            "parse_timeout_seconds",
            "max_pdf_pages",
            "max_image_pixels",
            "max_zip_entries",
            "max_zip_expanded_bytes",
            "max_zip_expansion_ratio",
            "max_extracted_bytes",
            "max_table_rows",
            "max_workbook_cells",
            "max_workbook_sheets",
        )
    }
    native_config["layout_enabled"] = profile["enabled"]
    native_config.update(options_dict(cfg))
    if "docling_artifacts_path" in parsing:
        native_config["docling_artifacts_path"] = parsing["docling_artifacts_path"]
    process = context.Process(target=_native_child, args=(child, filename, data, native_config))
    started = time.monotonic()
    try:
        process.start()
    except BaseException:
        parent.close()
        child.close()
        process.close()
        raise
    child.close()
    native_deadline = started + cfg.parse_timeout_seconds
    calls = 0
    reused = 0
    try:
        while True:
            if check_lease is not None:
                check_lease()
            remaining = native_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Native parsing exceeded its wall-time budget")
            if not parent.poll(min(remaining, 0.1)):
                if not process.is_alive():
                    raise UnsafeContentError("Native parser exited without a complete extraction")
                continue
            try:
                message = _receive(parent, native_deadline)
            except (EOFError, OSError, ValueError) as exc:
                raise UnsafeContentError(
                    "Native parser returned an invalid or incomplete result"
                ) from exc
            kind = message.get("kind")
            if kind == "error":
                raise UnsafeContentError(f"Native parsing failed ({message['error_type']})")
            if kind == "result":
                elements = [Element(**element) for element in message["elements"]]
                summary = message["summary"]
                summary.update(vision_calls=calls - reused, vision_reused=reused)
                save_artifact(result_key, {"elements": message["elements"], "summary": summary})
                return elements, summary
            if kind == "layout" and profile["enabled"]:
                image = base64.b64decode(message["image"], validate=True)
                key = hashlib.sha256(
                    json.dumps(
                        {"layout": profile, "image": hashlib.sha256(image).hexdigest()},
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                cached = load_artifact(key)
                regions = cached["regions"] if cached is not None else detect_layout(image, cfg)
                if check_lease is not None:
                    check_lease()
                if cached is None:
                    save_artifact(key, {"regions": regions})
                _send(parent, {"kind": "layout_result", "regions": regions}, native_deadline)
                continue
            if kind != "vision":
                raise UnsafeContentError("Native parser returned an unknown message")
            pause_started = time.monotonic()
            image = base64.b64decode(message["image"], validate=True)
            key = hashlib.sha256(
                json.dumps(
                    {
                        "image": hashlib.sha256(image).hexdigest(),
                        "prompt": message["prompt"],
                        "mime": message["mime"],
                        "model": gateway.vision_model,
                        "revision": PARSER_REVISION,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            calls += 1
            if calls > cfg.max_vision_calls:
                raise UnsafeContentError("Extraction exceeded the vision request budget")
            cached = load_artifact(key)
            if cached is not None:
                text = cached["text"]
                reused += 1
            else:
                if (calls - 1) * cfg.vision_max_tokens >= cfg.max_vision_tokens:
                    raise UnsafeContentError("Extraction exceeded the vision request budget")
                text = asyncio.run(
                    _vision_with_lease(
                        gateway,
                        image,
                        message["prompt"],
                        message["mime"],
                        cfg.vision_timeout_seconds,
                        min(
                            cfg.vision_max_tokens,
                            cfg.max_vision_tokens - (calls - 1) * cfg.vision_max_tokens,
                        ),
                        check_lease,
                    )
                )
                save_artifact(key, {"text": text})
            native_deadline += time.monotonic() - pause_started
            _send(parent, {"kind": "vision_result", "text": text}, native_deadline)
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        if process.is_alive():
            raise RuntimeError("Native parser could not be reaped")
        process.close()
