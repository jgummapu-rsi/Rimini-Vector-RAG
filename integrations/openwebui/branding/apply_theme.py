"""Best-effort static-asset patch applying the Rimini Street theme to an
installed `open-webui` pip package.

This is deliberately NOT the primary theming mechanism -- WEBUI_NAME,
WEBUI_FAVICON_URL (.env.openwebui) and the admin Banner (banner.html) are the
officially-supported knobs and always work. This script goes further (favicon
file replacement, a brand-color CSS overlay) by patching the installed
package's static build directory, because OSS Open WebUI gates full custom
theming behind an Enterprise license and offers no supported custom-CSS hook.

Safe by construction: every file it touches is backed up first (*.bak,
skipped if a backup already exists), changes are idempotent (a marker comment
prevents double-patching index.html), and it never fails loudly if Open
WebUI's build layout doesn't match what it expects in your installed version
-- it just reports what it did and did not find.

Run:  python integrations/openwebui/branding/apply_theme.py
Re-run after every `pip install --upgrade open-webui`.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

BRANDING_DIR = Path(__file__).resolve().parent
MARKER = "<!-- rimini-street-theme -->"


def _package_root() -> Path:
    spec = importlib.util.find_spec("open_webui")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("open-webui is not installed in this environment "
                  "(pip install open-webui) -- nothing to patch.")
    return Path(list(spec.submodule_search_locations)[0])


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"    backed up -> {bak.name}")


def _replace_favicons(root: Path) -> int:
    replaced = 0
    for name, src in (("favicon.ico", BRANDING_DIR / "favicon.ico"),
                       ("favicon.png", BRANDING_DIR / "favicon.png")):
        for hit in root.rglob(name):
            _backup(hit)
            shutil.copy2(src, hit)
            print(f"  favicon -> {hit}")
            replaced += 1
    return replaced


def _install_custom_css(root: Path) -> int:
    installed = 0
    for index_html in root.rglob("index.html"):
        text = index_html.read_text(encoding="utf-8", errors="ignore")
        if MARKER in text or "</head>" not in text:
            continue
        css_dest = index_html.parent / "rimini-custom.css"
        shutil.copy2(BRANDING_DIR / "custom.css", css_dest)
        link_tag = f'{MARKER}\n<link rel="stylesheet" href="/rimini-custom.css">\n</head>'
        _backup(index_html)
        index_html.write_text(text.replace("</head>", link_tag, 1), encoding="utf-8")
        print(f"  css injected -> {index_html} (+ {css_dest.name})")
        installed += 1
    return installed


def main() -> None:
    root = _package_root()
    print(f"open_webui package root: {root}")
    n_fav = _replace_favicons(root)
    n_css = _install_custom_css(root)

    if n_fav == 0 and n_css == 0:
        print("\nNothing matched -- this Open WebUI version's build layout differs "
              "from what this script expects. Inspect the package root above and "
              "adjust the rglob patterns in this file, or rely on the officially "
              "supported .env.openwebui + banner.html knobs instead.")
    else:
        print(f"\nDone: {n_fav} favicon file(s) replaced, {n_css} index.html "
              "patched with the brand CSS overlay.")
        print("Restart `open-webui serve` and hard-refresh the browser to see it.")
    print("Re-run this script after every `pip install --upgrade open-webui`.")


if __name__ == "__main__":
    main()
