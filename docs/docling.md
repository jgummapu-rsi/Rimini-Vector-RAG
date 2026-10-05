# Docling parsing

## Architecture

PDF and image uploads now use the vendored Docling 2.133.0 runtime by default:

`upload -> ingestion queue -> extract_bounded -> Docling -> Elements -> chunker
-> metadata -> embeddings -> pgvector/publication -> retrieval/citations`

The adapter runs inside the existing killable parser subprocess. It uses CPU
Heron layout inference, RapidOCR/ONNX, accurate TableFormer cell matching,
reading-order reconstruction and heading hierarchy inference. It preserves
item references, page numbers, top-left bounding boxes, section paths, tables
and source spans. Coordinates identify item regions rather than invented exact
word locations. Code items retain the chunker's atomic code-block route.

Figure crops pass through the existing cached, budgeted gateway vision channel.
This preserves diagram interpretation and table splitting without loading a
local generative vision model. Disable `DOCLING_PICTURE_DESCRIPTION` for fully
local PDF/image parsing; unprocessed pictures then appear as explicit omissions.
Formula/code generative enrichment is disabled for the CPU profile.

Office, spreadsheet, Markdown, HTML, CSV, RTF and text uploads keep their
specialized loaders, including spreadsheet cell coordinates and embedded-image
handling. `PARSING_BACKEND=native` selects the previous PDF/image route.
Docling PDFs do not invoke the separate DocLayout-YOLO model.

The parser cache identity includes upstream revision, model manifest, language,
OCR/table settings and resource limits. Partial/failed Docling conversions fail
the job; they are not published as complete evidence. Picture omissions are
reported through the existing partial-extraction summary.

## Install and provision

Use Linux and Python 3.12 or 3.13. From the repository root:

```bash
pip install -r requirements.txt
python -m scripts.download_docling
```

The model downloader uses `DOCLING_ARTIFACTS_PATH`, `DOCLING_OCR_LANGUAGE` and
`DOCLING_OCR_MODEL_SIZE` from Settings/environment. Re-run it after changing
language/model size. Provision before starting ingestion. The parser uses local
artifacts and disables Hugging Face downloads in its subprocess.

For development Compose, populate the named model volume using the same image
and runtime user as the worker:

```bash
docker compose --profile development build
docker compose --profile development run --rm --no-deps worker python -m scripts.download_docling
docker compose --profile development up -d
```

For production use the equivalent `worker-production` one-off command with the
production profile and release image before starting workers. Model downloads
need network access during provisioning. Model weights remain in the persistent
model volume, outside Git and the application source snapshot.

Existing `.env` overrides take precedence. Update old `PARSE_TIMEOUT_SECONDS=240`
and memory limits to the values below when deploying this branch.

## CPU sizing and quality

Suggested starting configuration, **not a measured benchmark**:

| Resource | Starting allocation |
| --- | --- |
| Worker | 4 CPU threads, one document at a time |
| Worker container RAM ceiling | 8 GiB |
| Full stack host | 4–8 CPU cores, 16 GiB RAM; 24–32 GiB for more headroom/concurrency |
| Parser virtual address-space limit | 16 GiB (not expected resident usage) |
| Model/runtime disk allowance | Reserve 5–10 GB initially; depends on installed wheels/artifacts |
| Parse wall timeout | 1,800 seconds/document, configurable |
| Stage batch size | 1; bounded queue of 2 |

The inspected WSL runtime had 12 logical CPUs (i7-1265U), 7.6 GiB RAM,
approximately 2.6 GiB available, and almost all of its 2 GiB swap occupied.
Increase WSL/container memory allocation before running the full stack.

For capacity planning only, allow roughly 1–10 seconds/page for typical digital
PDFs and 5–30+ seconds/page for scans or complex tables on a modern CPU. These
are broad estimates, not Docling measurements on this machine. At those rates,
100 pages can take roughly 2–17 minutes or 8–50+ minutes respectively. The
default timeout may need raising for large scanned documents. Cold subprocess
startup loads models for each uncached document; model downloads are separate.

End-to-end ingestion also includes figure gateway latency, metadata extraction,
embedding requests and database writes. Gateway embeddings/chat are the existing
defaults, so the whole RAG stack is not fully local CPU inference. Time is roughly
model load + page conversion + figure requests + metadata/embedding + publication.
Adding CPU threads does not guarantee proportional speedup; parallel workers
duplicate model memory. The CPU-time limit accounts for configured thread count,
while the parent independently enforces elapsed time and lease cancellation.

Quality defaults prioritize table structure and reading order. Native PDF text
is retained where available and OCR covers image regions. Use
`DOCLING_FORCE_FULL_PAGE_OCR=true` when embedded text is corrupted; it costs more
and can reduce quality on otherwise clean digital PDFs. Choose the appropriate
single OCR language/code (`en` by default). Accurate extraction still depends on
source quality: handwriting, tiny text, complex formulas and ambiguous tables
need corpus-specific evaluation before claiming an accuracy improvement.

## Source audit and verification status

The upstream converter imports multiple backend/pipeline classes eagerly, and
default plugin discovery imports additional model classes. Therefore the vendor
snapshot contains their transitive import closure, including optional branches,
rather than deleting modules merely because their inference feature is disabled.
See `vendor/docling/UPSTREAM.json` and `scripts/vendor_docling.py`. Upstream MIT
notices are retained. Only CPU PDF/OCR dependencies are selected in the local
package metadata; no upstream CLI or VLM extras are installed.

Application pipeline, process isolation, chunking/provenance consumers, upstream
converter, pipeline options, plugin registration, OCR, layout/table stages and
model artifact handling were reviewed. Structural indexing covered the app's
106 code files; this is not a claim of a line-by-line review of every upstream
file. Tests, inference benchmarks and model downloads were intentionally not run
for this integration, as requested. Runtime correctness and throughput remain
unverified until deployment validation is performed.
