# Vendored Docling CPU runtime

Source: <https://github.com/docling-project/docling.git>

Version **2.133.0**, commit **caba660f6ff3a2ee8df39bd39d478046749f815c**.
See `LICENSE`, `UPSTREAM_README.md`, and `UPSTREAM.json` (per-file SHA-256).

`scripts/vendor_docling.py` exports the transitive Python import closure of
`document_converter`, default plugin registration, and model downloading. It
includes imports in optional branches: deleting them breaks upstream imports or
plugin discovery even when only PDF/image conversion is selected. Runtime
resources are retained. Upstream source files are unmodified.

Tests, datasets, examples, documentation site, notebooks, CLI, service client,
standalone extraction/chunking entry points and Git history are excluded unless
required by that import closure. The local `pyproject.toml` preserves the upstream
distribution name/version and plugin entry point, selecting only CPU PDF/OCR
dependencies. Dependencies such as docling-core, docling-parse and model runtimes
remain installed packages; weights are provisioned separately.

Install from the **repository root** with `pip install -r requirements.txt`.
Do not add this directory to `sys.path`: installation supplies package metadata
and plugin discovery. See `docs/docling.md` for configuration and provisioning.

To refresh deliberately, update the pinned revision/version in the export script
and package metadata, clone upstream, checkout that revision, export into a clean
vendor directory, and review the resulting manifest and dependency changes.
