"""Local layout inference. No OCR and no downloads on the ingestion path."""

from __future__ import annotations

import hashlib
import io
import math
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from PIL import Image

from scripts.download_doclayout import SHA256

_INFERENCE_LOCK = Lock()


def layout_profile(cfg) -> dict:
    if not getattr(cfg, "layout_enabled", False):
        return {"enabled": False}
    path = Path(cfg.layout_model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Layout checkpoint missing: {path}. Run python -m scripts.download_doclayout"
        )
    stat = path.stat()
    digest = _checkpoint_digest(str(path), stat.st_size, stat.st_mtime_ns)
    if digest != SHA256:
        raise ValueError("Layout checkpoint does not match the pinned DocStructBench model")
    return {
        "enabled": True,
        "sha256": digest,
        "package": "doclayout-yolo==0.0.4",
        "imgsz": cfg.layout_image_size,
        "confidence": cfg.layout_confidence,
        "device": cfg.layout_device,
    }


@lru_cache(maxsize=4)
def _checkpoint_digest(path: str, size: int, modified: int) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _model(path: str, digest: str):
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/doclayout-yolo")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/doclayout-matplotlib")
    # Keep Torch out of the memory-limited parser forkserver and initialize its
    # writable configuration directories before loading the inference runtime.
    from doclayout_yolo import YOLOv10  # noqa: PLC0415

    return YOLOv10(path)


def detect_layout(image_bytes: bytes, cfg) -> list[dict]:
    """Return top-left normalized boxes in the original rendered image space."""

    profile = layout_profile(cfg)
    if not profile["enabled"]:
        return []
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.width * image.height > cfg.max_image_pixels:
            raise ValueError("Layout page exceeds pixel limit")
        with _INFERENCE_LOCK:
            model = _model(str(Path(cfg.layout_model_path).resolve()), profile["sha256"])
            result = model.predict(
                image.convert("RGB"),
                imgsz=cfg.layout_image_size,
                conf=cfg.layout_confidence,
                device=cfg.layout_device,
                verbose=False,
                save=False,
                max_det=300,
            )[0]
        height, width = result.orig_shape
        regions = []
        for box in result.boxes.data.cpu().tolist():
            x0, y0, x1, y1, confidence, label = box
            if not all(math.isfinite(v) for v in box):
                continue
            x0, x1 = max(0, x0 / width), min(1, x1 / width)
            y0, y1 = max(0, y0 / height), min(1, y1 / height)
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "label": result.names[int(label)],
                    "confidence": confidence,
                }
            )
    regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0], region["label"]))
    return [dict(region, id=index) for index, region in enumerate(regions)]
