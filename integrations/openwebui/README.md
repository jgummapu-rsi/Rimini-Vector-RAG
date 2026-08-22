# Rimini Street Knowledge Assistant — Open WebUI integration

Turns Open WebUI into the chat front end for this repo's RAG pipeline, in place of
testing through the FastAPI Swagger UI. One Pipe function does both jobs:

- **Attach a file** → it's forwarded to `POST /ingest` and you watch it flow
  stage-by-stage (parse → route → extract → chunk → metadata → embed → binarize →
  index) via live status updates, until it's searchable.
- **Ask a question** → it's forwarded to `POST /query` and you see the retrieval
  pipeline actually taken (decompose → hybrid retrieve → rerank → generate),
  followed by the answer with numbered citations back to the source
  document/section.

No separate worker process, no Docker — the Pipe is a single Python file pasted
into Open WebUI's Admin Panel, calling this repo's existing API over HTTP.

## 1. Start the RAG backend (as usual)

```bash
source .venv/Scripts/activate
uvicorn app.api.app:app --port 8000        # terminal 1
python -m app.worker                        # terminal 2 -- must be running to process jobs
python -m scripts.seed                      # prints a bearer token, e.g. sk-...
```

Keep the printed `API token` — it's `RAG_API_TOKEN` below.

## 2. Install and run Open WebUI

Open WebUI currently requires **Python >=3.11 and <3.13**. Use a *separate*
virtual environment from this repo's `.venv` — Open WebUI has its own large,
independent dependency tree (its own torch/transformers etc.), and if your
machine's default Python is 3.13+ (verified 3.13.14 on this machine while
building this integration — `pip install open-webui` fails outright there
with "no matching distribution"), install a 3.11 or 3.12 interpreter first
(python.org, or `pyenv`/`pyenv-win`) and point the new venv at it explicitly:

```bash
# example using the Windows 'py' launcher once a 3.11/3.12 interpreter is installed
py -3.12 -m venv .venv-openwebui
source .venv-openwebui/Scripts/activate
pip install open-webui
open-webui serve                            # defaults to http://localhost:8080
```

Log in (first account created becomes admin), then create an Open WebUI API key
for the Pipe to use when downloading chat attachments:
**Settings → Account → API Keys → Create new key.**

## 3. Install the Pipe

**Admin Panel → Functions → + (Create Function)** → paste the contents of
[`rimini_rag_pipe.py`](./rimini_rag_pipe.py) → Save → enable it.

Open the function's **Valves** and set:

| Valve | Value |
|---|---|
| `RAG_BASE_URL` | `http://localhost:8000` (the FastAPI service) |
| `RAG_API_TOKEN` | the token from `scripts.seed` above |
| `OPENWEBUI_BASE_URL` | `http://localhost:8080` (Open WebUI itself) |
| `OPENWEBUI_API_KEY` | the API key created in step 2 |
| `TOP_K` | 5 (default) |
| `POLL_INTERVAL_S` / `POLL_TIMEOUT_S` | 1.5 / 180 (default) |

Select **"Rimini Street Knowledge Assistant"** as the model in a new chat.

## 4. Apply the Rimini Street theme

See [`branding/README.md`](./branding/README.md).

## 5. Try it

1. Attach a PDF from `sample_pdfs/` and send. Watch the status trail tick through
   every real pipeline stage to "indexed and ready".
2. Ask a question about its content. Watch the retrieval trace appear, then the
   citations (source document + page/section), then the answer.

## How it works

`rimini_rag_pipe.py` is a single Open WebUI **Pipe function** (the current,
in-process extension mechanism — the older separate "Pipelines" worker is
deprecated upstream). It never imports anything from `app/*`; it only speaks
HTTP to the FastAPI service, so the ports/adapters boundary in the main
`CLAUDE.md` is respected even though this is client-side code.

- Attachment bytes are fetched via Open WebUI's own public REST API
  (`GET /api/v1/files/{id}/content`), not internal storage classes, since those
  vary across Open WebUI releases — the public API is the stable contract.
- Only the current turn's attachments are processed (not the whole
  conversation's) to avoid re-ingesting old files on every follow-up question;
  this repo's `/ingest` is already sha256-idempotent, so this is a UX
  optimization, not a safety requirement.
- Pipeline stage progress and the retrieval trace are surfaced via Open WebUI's
  `status` event; citations via its `citation` event — both render natively in
  the chat UI as a collapsible trace above the answer, no custom frontend code
  needed.

## Known limitations

- `branding/apply_theme.py`'s file-search patterns could not be verified
  against a live `open-webui` install while building this integration (this
  machine only has Python 3.13, which `open-webui` doesn't yet support — see
  above). Run it once after your first `open-webui serve` and check its
  output; if it reports "nothing matched", see
  [`branding/README.md`](./branding/README.md) for how to adjust it. The
  officially-supported `.env.openwebui` + `banner.html` knobs need no such
  verification and always work.
- If the Open WebUI attachment message shape changes in a future release,
  `_current_turn_attachments` may need its `messages[-1]["files"]` lookup
  adjusted — verify against your installed version if attachments stop being
  detected.
- A crashed/restarted worker mid-ingest will show as "still processing" once
  `POLL_TIMEOUT_S` elapses; this mirrors the backend's own known gap (no
  stuck-job reaper yet, see the main `CLAUDE.md` roadmap) rather than hiding it.
