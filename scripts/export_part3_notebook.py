"""Export the executed Part 3 notebook, retaining code, prose and cell outputs.

Example (from the project root):
    .part3_env/Scripts/python.exe scripts/export_part3_notebook.py

This script does not execute or modify the source notebook. It prints the
source SHA256 so the exported artifacts can be tied to that saved run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
from traitlets.config import Config


ROOT = Path(__file__).resolve().parents[1]
PRINT_CSS = r"""
<style id="part3-print-layout">
@page { size: A4; margin: 15mm 13mm 16mm 13mm; }
html, body {
  color: #111 !important; background: white !important;
  width: auto !important; min-width: 0 !important; max-width: 100% !important;
  margin: 0 !important; padding: 0 !important;
  font-size: 10pt; -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
main, #notebook, #notebook-container, .jp-Notebook {
  width: auto !important; min-width: 0 !important; max-width: 100% !important;
  margin: 0 !important; padding: 0 !important; box-shadow: none !important;
}
.jp-Cell, .cell, .jp-InputArea, .jp-OutputArea, .jp-OutputArea-child,
.jp-InputArea-editor, .input_area, .output_area, .output_subarea,
.jp-OutputArea-output, .jp-RenderedText, .jp-RenderedHTMLCommon,
.jp-CodeMirrorEditor, .cm-editor, .cm-scroller, .highlight {
  min-width: 0 !important; max-width: 100% !important;
  max-height: none !important; height: auto !important;
  overflow: visible !important; text-overflow: clip !important;
}
.jp-Cell, .cell { break-inside: auto !important; margin: 0 0 3mm !important; }
.jp-Cell-inputWrapper, .jp-Cell-outputWrapper {
  display: block !important; overflow: visible !important;
}
.jp-InputArea-prompt, .jp-OutputArea-prompt, .prompt {
  min-width: 10mm !important; width: 10mm !important;
  padding-left: 0 !important; font-size: 7pt !important;
  flex: 0 0 10mm !important;
}
.jp-InputArea, .jp-OutputArea-child { display: flex !important; }
.jp-InputArea-editor, .jp-OutputArea-output { flex: 1 1 0 !important; }
pre, code, .highlight pre, .jp-RenderedText pre {
  white-space: pre-wrap !important; overflow-wrap: anywhere !important;
  word-break: break-word !important; tab-size: 4;
  max-width: 100% !important; max-height: none !important;
  overflow: visible !important; text-overflow: clip !important;
  font-size: 8pt !important; line-height: 1.32 !important;
}
pre { break-inside: auto !important; }
.jp-RenderedHTMLCommon { font-size: 10pt !important; line-height: 1.42; }
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li {
  overflow-wrap: anywhere; orphans: 3; widows: 3;
}
h1, h2, h3, h4 { break-after: avoid-page; page-break-after: avoid; }
h1 { font-size: 20pt !important; }
h2 { font-size: 15pt !important; margin-top: 5mm !important; }
h3 { font-size: 12pt !important; }
.jp-RenderedImage, .output_png, .output_svg,
.jp-OutputArea-child:has(.jp-RenderedImage) {
  break-inside: avoid-page !important; page-break-inside: avoid !important;
}
img, svg {
  max-width: 100% !important; max-height: 230mm !important;
  width: auto; height: auto; object-fit: contain;
}
table { max-width: 100% !important; font-size: 8pt !important; border-collapse: collapse; }
td, th { overflow-wrap: anywhere; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
.anchor-link, .jp-Cell-inputCollapser, .jp-Cell-outputCollapser { display: none !important; }
@media print {
  a { color: #174875 !important; text-decoration: none; }
  .jp-Notebook { overflow: visible !important; }
  .jp-OutputArea-output { display: block !important; }
}
</style>
"""


def find_browser(explicit: Path | None) -> Path | None:
    """Prefer an explicit executable, then an existing local headless shell."""
    if explicit is not None:
        browser = explicit.expanduser().resolve()
        if not browser.is_file():
            raise FileNotFoundError(f"Browser executable does not exist: {browser}")
        return browser
    override = os.environ.get("PART3_BROWSER_EXECUTABLE")
    if override:
        return find_browser(Path(override))
    # Packaged Windows apps can redirect LOCALAPPDATA to a per-app directory.
    # Also check the real user cache where Playwright was previously installed.
    caches = {Path.home() / "AppData/Local/ms-playwright"}
    if os.environ.get("LOCALAPPDATA"):
        caches.add(Path(os.environ["LOCALAPPDATA"]) / "ms-playwright")
    candidates = [path for cache in caches for path in cache.glob(
        "chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe"
    )]
    if candidates:
        return sorted(candidates, key=lambda p: p.parents[1].name, reverse=True)[0]
    # None lets Playwright resolve its normal browser cache on other platforms.
    return None


def validate_executed_notebook(notebook):
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    if not code_cells:
        raise ValueError("The notebook contains no executable code cells.")
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        if cell.execution_count is None:
            raise ValueError(f"Cell {index} has not been executed; export an executed notebook.")
        errors = [output for output in cell.outputs if output.output_type == "error"]
        if errors:
            raise ValueError(f"Cell {index} contains an error output: {errors[0].get('ename', 'error')}")
    output_count = sum(len(c.get("outputs", [])) for c in code_cells)
    if output_count == 0:
        raise ValueError("The notebook has no saved cell outputs.")
    image_count = sum("image/png" in output.get("data", {})
                      for cell in code_cells for output in cell.get("outputs", []))
    if image_count != 5:
        raise ValueError(f"Expected all five Part 3 figures embedded in outputs, found {image_count}.")
    return len(code_cells), output_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "part3_pythia.ipynb")
    parser.add_argument("--html", type=Path, default=ROOT / "outputs/part3/part3_notebook.html")
    parser.add_argument("--pdf", type=Path, default=ROOT / "report/part3_notebook.pdf")
    parser.add_argument("--browser-executable", type=Path)
    parser.add_argument("--mathjax-url", help="Optional local/remote MathJax URL for offline or custom setups.")
    args = parser.parse_args()
    source = args.notebook.resolve()
    notebook = nbformat.read(source, as_version=4)
    code_count, output_count = validate_executed_notebook(notebook)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    config = Config()
    config.HTMLExporter.exclude_input = False
    config.HTMLExporter.exclude_output = False
    config.HTMLExporter.exclude_input_prompt = False
    config.HTMLExporter.exclude_output_prompt = False
    config.HTMLExporter.exclude_markdown = False
    config.HTMLExporter.embed_images = True
    exporter = HTMLExporter(config=config, template_name="lab")
    if args.mathjax_url:
        exporter.mathjax_url = args.mathjax_url
    html, _ = exporter.from_notebook_node(
        notebook, resources={"metadata": {"path": str(source.parent), "name": source.stem}}
    )
    metadata = f'<meta name="source-notebook-sha256" content="{digest}">'
    html = html.replace("</head>", metadata + PRINT_CSS + "\n</head>", 1)
    html_path, pdf_path = args.html.resolve(), args.pdf.resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    executable = find_browser(args.browser_executable)
    with sync_playwright() as playwright:
        launch_args = {"headless": True}
        if executable is not None:
            launch_args["executable_path"] = str(executable)
        browser = playwright.chromium.launch(**launch_args)
        try:
            page = browser.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=1)
            page.goto(html_path.as_uri(), wait_until="load", timeout=60000)
            # nbconvert's lab template normally uses MathJax 2; also support v3.
            page.evaluate("""async () => {
                await document.fonts.ready;
                if (window.MathJax && MathJax.Hub && MathJax.Hub.Queue) {
                    await new Promise(resolve => MathJax.Hub.Queue(resolve));
                } else if (window.MathJax && MathJax.typesetPromise) {
                    await MathJax.typesetPromise();
                }
                await Promise.all(Array.from(document.images).map(img =>
                    img.complete ? Promise.resolve() : new Promise((resolve, reject) => {
                        img.onload = resolve; img.onerror = reject;
                    })));
            }""")
            broken_images = page.evaluate("() => Array.from(document.images).filter(i => !i.complete || i.naturalWidth === 0).length")
            if broken_images:
                raise RuntimeError(f"Cannot export: {broken_images} notebook images failed to load.")
            page.emulate_media(media="print")
            page.pdf(
                path=str(pdf_path), format="A4", prefer_css_page_size=True,
                print_background=True, display_header_footer=True,
                header_template="<span></span>",
                footer_template=(
                    '<div style="font-size:8px;color:#666;width:100%;text-align:center;">'
                    'CSE256 PA2 - Part B &nbsp; | &nbsp; '
                    '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
                ),
                margin={"top": "15mm", "right": "13mm", "bottom": "16mm", "left": "13mm"},
            )
        finally:
            browser.close()

    print(f"Notebook: {source}")
    print(f"Source SHA256: {digest}")
    print(f"Preserved {len(notebook.cells)} cells, {code_count} nonempty code cells, {output_count} saved outputs.")
    print(f"HTML: {html_path}")
    print(f"PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
