"""Guards against resource exhaustion from malicious or malformed uploads:
image/decompression-bomb checks, table row caps, and a soft per-parse timeout."""
from __future__ import annotations

import io
import threading
from typing import Any, Callable


class UnsafeContentError(ValueError):
    """A file exceeds a configured safety limit (page count, image size, row count)."""


def check_image_pixels(data: bytes, max_pixels: int) -> None:
    """Raise UnsafeContentError if the image's reported width*height exceeds
    `max_pixels`, without fully decoding the pixel data first -- PIL's `Image.open`
    reads only the header, so an oversized image is rejected before the
    (potentially memory-exhausting) decode that reading pixels would trigger."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception as e:
        raise UnsafeContentError(f"unreadable image: {e}") from e
    pixels = width * height
    if pixels > max_pixels:
        raise UnsafeContentError(
            f"image is {width}x{height} ({pixels:,} px), exceeds the "
            f"{max_pixels:,} px limit"
        )


def check_row_count(row_count: int, max_rows: int, kind: str = "table") -> None:
    """Raise UnsafeContentError if a parsed table has more than `max_rows` rows."""
    if row_count > max_rows:
        raise UnsafeContentError(
            f"{kind} has {row_count:,} rows, exceeds the {max_rows:,} row limit"
        )


def run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run `fn` (no-arg callable) on a worker thread and wait up to
    `timeout_seconds`. Raises TimeoutError if it hasn't finished in time.

    This is a SOFT timeout: Python has no safe way to forcibly kill a thread,
    so a timed-out call's thread keeps running in the background (daemonized,
    so it won't block process exit) -- the point is bounding how long ONE bad
    document can block the worker from moving on to the next job, not
    guaranteeing the abandoned work stops immediately.
    """
    result: dict[str, Any] = {}

    def _target():
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread below
            result["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"exceeded {timeout_seconds}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")
