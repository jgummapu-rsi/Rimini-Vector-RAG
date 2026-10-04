from __future__ import annotations

import io
import zipfile

from PIL import Image


class UnsafeContentError(ValueError):
    """A file exceeds a configured safety limit (page count, image size, row count)."""


def check_archive_expansion(data: bytes, filename: str, cfg) -> None:
    if not filename.lower().endswith((".docx", ".xlsx")):
        return
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > cfg.max_zip_entries:
                raise UnsafeContentError("Office archive exceeds the entry limit")
            total = 0
            for entry in entries:
                if entry.flag_bits & 1:
                    raise UnsafeContentError("Encrypted office archives are unsupported")
                total += entry.file_size
                if total > cfg.max_zip_expanded_bytes:
                    raise UnsafeContentError("Office archive exceeds the expanded-byte limit")
                if entry.file_size > max(1, entry.compress_size) * cfg.max_zip_expansion_ratio:
                    raise UnsafeContentError("Office archive exceeds the expansion-ratio limit")
                actual = 0
                with archive.open(entry) as member:
                    while block := member.read(64 * 1024):
                        actual += len(block)
                        if actual > entry.file_size:
                            raise UnsafeContentError(
                                "Office archive has inconsistent expanded size"
                            )
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise UnsafeContentError("Malformed office archive") from exc


def check_image_pixels(data: bytes, max_pixels: int) -> None:
    """Raise UnsafeContentError if the image's reported width*height exceeds
    `max_pixels`, without fully decoding the pixel data first -- PIL's `Image.open`
    reads only the header, so an oversized image is rejected before the
    (potentially memory-exhausting) decode that reading pixels would trigger."""

    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception as e:
        raise UnsafeContentError(f"unreadable image: {e}") from e
    pixels = width * height
    if pixels > max_pixels:
        raise UnsafeContentError(
            f"image is {width}x{height} ({pixels:,} px), exceeds the {max_pixels:,} px limit"
        )
