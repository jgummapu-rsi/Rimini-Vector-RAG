"""Provision just the CPU parser models before ingestion (no inference).

Run from the repository root: python -m scripts.download_docling
Uses the same Settings as the worker; records package versions and hashes so
changing model artifacts invalidates cached extraction results.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from app.ingest.pipeline.docling_config import DOCLING_REVISION
from app.shared.config import settings


def main() -> None:
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions  # noqa: PLC0415
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel  # noqa: PLC0415
    from docling.models.stages.table_structure.table_structure_model import (  # noqa: PLC0415
        TableStructureModel,
    )
    from docling.models.utils.hf_model_download import download_hf_model  # noqa: PLC0415

    root = Path(settings.docling_artifacts_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    # Incomplete provisioning must not look ready to a new worker.
    manifest_path.unlink(missing_ok=True)
    spec = LayoutObjectDetectionOptions().model_spec
    download_hf_model(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=root / spec.repo_id.replace("/", "--"),
        progress=True,
    )
    TableStructureModel.download_models(
        local_dir=root / TableStructureModel._model_repo_folder,
        progress=True,
    )
    RapidOcrModel.download_models(
        backend="onnxruntime",
        lang=settings.docling_ocr_language,
        model_size=settings.docling_ocr_model_size,
        local_dir=root / RapidOcrModel._model_repo_folder,
        progress=True,
    )
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.name != "manifest.json":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[str(path.relative_to(root))] = digest
    manifest = {
        "docling_revision": DOCLING_REVISION,
        "ocr_language": settings.docling_ocr_language,
        "ocr_model_size": settings.docling_ocr_model_size,
        "packages": {
            name: version(name)
            for name in (
                "docling-slim",
                "docling-core",
                "docling-parse",
                "docling-ibm-models",
                "rapidocr",
                "torch",
                "transformers",
                "onnxruntime",
            )
        },
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Docling CPU artifacts ready: {root} ({len(files)} files)")


if __name__ == "__main__":
    main()
