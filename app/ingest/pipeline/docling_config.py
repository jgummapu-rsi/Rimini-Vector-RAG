"""Lightweight Docling identity/configuration shared by parent and parser child."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DOCLING_REVISION = "caba660f6ff3a2ee8df39bd39d478046749f815c"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}
DEFAULTS = {
    "parsing_backend": "docling",
    "docling_artifacts_path": "models/docling",
    "docling_num_threads": 4,
    "docling_ocr_language": "en",
    "docling_ocr_model_size": "small",
    "docling_force_full_page_ocr": False,
    "docling_table_mode": "accurate",
    "docling_picture_description": True,
    "docling_batch_size": 1,
}


def uses_docling(filename: str, cfg) -> bool:
    return getattr(cfg, "parsing_backend", "docling") == "docling" and Path(
        filename
    ).suffix.lower() in {".pdf", *IMAGE_EXTENSIONS}


def options_dict(cfg) -> dict:
    return {key: getattr(cfg, key, default) for key, default in DEFAULTS.items()}


def parser_profile(filename: str, cfg) -> dict:
    if not uses_docling(filename, cfg):
        return {"backend": "native"}
    options = options_dict(cfg)
    path = Path(options["docling_artifacts_path"]).resolve()
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Docling artifacts missing at {path}. Run python -m scripts.download_docling"
        )
    manifest_bytes = manifest.read_bytes()
    identity = json.loads(manifest_bytes)
    expected = {
        "docling_revision": DOCLING_REVISION,
        "ocr_language": options["docling_ocr_language"],
        "ocr_model_size": options["docling_ocr_model_size"],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Docling artifacts do not match configuration; re-run scripts.download_docling"
        )
    return {
        **options,
        "docling_artifacts_path": str(path),
        "revision": DOCLING_REVISION,
        "artifacts_manifest": hashlib.sha256(manifest_bytes).hexdigest(),
    }
