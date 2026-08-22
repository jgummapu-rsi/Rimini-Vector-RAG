# Rimini Street branding for Open WebUI

Brand colors sampled directly from `sample_pdfs/Logo.png`: background
**`#FBE232`**, wordmark/rule **`#0A0A0A`**.

Three layers, in order of how reliably they'll survive an Open WebUI upgrade:

## 1. Officially supported (always works) — `.env.openwebui`

```bash
set -a && source integrations/openwebui/branding/.env.openwebui && set +a
open-webui serve
```

Sets `WEBUI_NAME` (browser tab / app title) and `WEBUI_FAVICON_URL` (tab icon,
generated from the logo as `favicon.png`/`favicon.ico` in this directory).

## 2. Officially supported (always works) — the admin Banner

**Admin Panel → Settings → General → Banners → +** → paste the contents of
[`banner.html`](./banner.html) into the Content field. Renders a yellow
brand strip with black text/rule above every logged-in user's chat, using raw
HTML (Open WebUI banners don't render Markdown).

## 3. Best effort (may need adjustment per version) — `apply_theme.py`

Full theming (custom accent colors app-wide, logo replacing Open WebUI's own
in the sidebar) is **gated behind an Open WebUI Enterprise license** in the
OSS build — there's no supported hook for it today. `apply_theme.py` patches
the installed `open-webui` pip package's static build directory directly:
replaces its favicon files and injects `custom.css` (the brand palette) into
any `index.html` it finds.

```bash
python integrations/openwebui/branding/apply_theme.py
```

- Backs up every file it touches (`*.bak`) before changing it.
- Idempotent — safe to re-run, and required after every
  `pip install --upgrade open-webui` (upgrades overwrite the patched files).
- If it reports "nothing matched", your installed version's build layout
  differs from what the script's search patterns expect — open the printed
  package root in a file browser, find the actual `index.html`/`favicon.*`
  paths, and adjust `_replace_favicons`/`_install_custom_css` in
  `apply_theme.py` accordingly. Layers 1 and 2 above are unaffected either way.

`custom.css` itself documents which selectors are safe bets (scrollbars, a
handful of CSS variables recent Open WebUI versions expose) versus which are
genuinely best-effort — read its header comment before assuming a specific
element will get skinned.
