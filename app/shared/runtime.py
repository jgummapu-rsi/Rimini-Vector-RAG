"""Native runtime bootstrap.

This Windows environment lacks the MSVC C++ runtime (msvcp140.dll etc.), which
onnxruntime needs. The `msvc-runtime` pip package drops those DLLs into the venv;
we just have to make that directory discoverable before importing onnxruntime.
Call `ensure_native_runtime()` before any onnxruntime import.
"""
from __future__ import annotations

import glob
import os
import sys

_done = False


def ensure_native_runtime() -> None:
    """Make the MSVC runtime DLLs discoverable, once per process, before onnxruntime is imported."""
    global _done
    if _done:
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    for d in (sys.prefix, os.path.join(sys.prefix, "Scripts")):
        try:
            if glob.glob(os.path.join(d, "msvcp140.dll")):
                os.add_dll_directory(d)
        except Exception:
            pass
    _done = True
