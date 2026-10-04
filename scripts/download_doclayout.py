"""Download and verify the single DocStructBench checkpoint, without cloning code."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
SHA256 = "9a2ee0220fe3d9ad31b47e1d9f1282f46959a54e4618fce9cffcc9715b8286e2"
URL = f"https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/{FILENAME}"


def download(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / FILENAME
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"Existing checkpoint has an unexpected hash: {destination}")
        return destination
    temporary = destination.with_suffix(".part")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != SHA256:
            raise ValueError("Downloaded checkpoint failed SHA-256 verification")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("models/doclayout-yolo"))
    args = parser.parse_args()
    print(f"Verified checkpoint: {download(args.directory)}")
