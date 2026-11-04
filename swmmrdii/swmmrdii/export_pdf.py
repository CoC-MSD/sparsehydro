"""
Export a Jupyter notebook to PDF without LaTeX.

Pipeline: notebook -> HTML (via nbconvert) -> PDF (via headless Chrome/Edge).
This correctly renders interactive Plotly figures, which the LaTeX-based
``nbconvert --to pdf`` exporter cannot do.

Usage
-----
From a terminal (in the sparsehydro env)::

    python -m swmmrdii.export_pdf "path/to/notebook.ipynb"
    python -m swmmrdii.export_pdf "path/to/notebook.ipynb" -o out.pdf

Or from a notebook cell::

    from swmmrdii.export_pdf import notebook_to_pdf
    notebook_to_pdf('LC_sparsehydro_03145483_from_csv_swmmrdii.ipynb')

Requirements (already present in the ArcGIS sparsehydro env):
    * nbconvert
    * Google Chrome or Microsoft Edge installed (auto-detected)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


# Common install locations for Chromium-based browsers on Windows.
_BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find_browser() -> str:
    """Locate a Chromium-based browser executable, or raise."""
    # Honour an explicit override first.
    env = os.environ.get("SWMMRDII_BROWSER")
    if env and os.path.exists(env):
        return env
    for path in _BROWSER_CANDIDATES:
        if os.path.exists(path):
            return path
    # Fall back to anything on PATH.
    for name in ("chrome", "chrome.exe", "msedge", "msedge.exe", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "No Chrome/Edge/Chromium browser found. Install one, or set the "
        "SWMMRDII_BROWSER environment variable to the executable path."
    )


def notebook_to_pdf(
    nb_path: str,
    out_pdf: str | None = None,
    render_delay_ms: int = 4000,
    keep_html: bool = False,
) -> str:
    """
    Convert *nb_path* to a PDF and return the output path.

    Parameters
    ----------
    nb_path : str
        Path to the .ipynb file.  Its saved cell outputs (including rendered
        Plotly figures) are exported as-is — re-run the notebook first if you
        want fresh results.
    out_pdf : str or None
        Output PDF path.  Defaults to the notebook path with a .pdf extension.
    render_delay_ms : int
        Milliseconds Chrome waits after loading the HTML before printing, so
        the Plotly JavaScript has time to draw the figures.  Increase for very
        large notebooks (default 4000).
    keep_html : bool
        If True, keep the intermediate HTML file next to the PDF (useful for
        debugging).  Default False.

    Returns
    -------
    str
        Path to the generated PDF.
    """
    nb_path = os.path.abspath(nb_path)
    if not os.path.exists(nb_path):
        raise FileNotFoundError(nb_path)

    if out_pdf is None:
        out_pdf = os.path.splitext(nb_path)[0] + ".pdf"
    out_pdf = os.path.abspath(out_pdf)

    browser = _find_browser()

    # --- Step 1: notebook -> HTML (embeds Plotly JS via the default template) -
    nb_dir = os.path.dirname(nb_path)
    base = os.path.splitext(os.path.basename(nb_path))[0]
    html_path = os.path.join(nb_dir, base + ".html")

    print(f"[1/2] nbconvert -> HTML: {html_path}")
    subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--output", base,        # writes <base>.html in nb_dir
            "--output-dir", nb_dir,
            nb_path,
        ],
        check=True,
    )

    # --- Step 2: HTML -> PDF (headless Chrome) --------------------------------
    # file:// URL so Chrome can load local resources.
    file_url = "file:///" + html_path.replace("\\", "/")
    # A dedicated temp profile avoids touching the user's real browser session.
    profile = tempfile.mkdtemp(prefix="swmmrdii_chrome_")

    print(f"[2/2] {os.path.basename(browser)} --headless -> PDF: {out_pdf}")
    try:
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--virtual-time-budget={render_delay_ms}",
                f"--print-to-pdf={out_pdf}",
                file_url,
            ],
            check=True,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)
        if not keep_html:
            try:
                os.remove(html_path)
            except OSError:
                pass

    if not os.path.exists(out_pdf):
        raise RuntimeError(
            "PDF was not produced. Try increasing render_delay_ms, or run with "
            "keep_html=True and open the HTML to confirm the figures render."
        )
    print(f"Done -> {out_pdf}")
    return out_pdf


def _main() -> None:
    p = argparse.ArgumentParser(description="Export a notebook to PDF (no LaTeX).")
    p.add_argument("notebook", help="Path to the .ipynb file.")
    p.add_argument("-o", "--output", default=None, help="Output PDF path.")
    p.add_argument(
        "--delay", type=int, default=4000,
        help="Chrome render delay in ms before printing (default 4000).",
    )
    p.add_argument("--keep-html", action="store_true", help="Keep the intermediate HTML.")
    args = p.parse_args()
    notebook_to_pdf(
        args.notebook,
        out_pdf=args.output,
        render_delay_ms=args.delay,
        keep_html=args.keep_html,
    )


if __name__ == "__main__":
    _main()
