"""``sparsehydro-viewer`` console script: launches the Streamlit app.

Usage::

    sparsehydro-viewer [path/to/rain_stormflow.csv] [--server.port 8502 ...]

A leading positional CSV path is forwarded to the app as its default dataset;
any remaining arguments are passed through to ``streamlit run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")


def build_argv(args: "list[str]") -> "list[str]":
    """Translate ``sparsehydro-viewer`` arguments into a ``streamlit run`` argv.

    :param args: Command-line arguments (without the program name).
    :type args: list[str]
    :returns: Argument vector for :func:`streamlit.web.cli.main`.
    :rtype: list[str]
    """
    app_args: list[str] = []
    if args and not args[0].startswith("-"):
        app_args = [str(Path(args[0]).resolve())]
        args = args[1:]
    argv = ["streamlit", "run", str(APP_PATH), *args]
    if app_args:
        argv += ["--", *app_args]
    return argv


def main() -> None:
    """Entry point for the ``sparsehydro-viewer`` console script."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:  # pragma: no cover - depends on environment
        sys.exit("The viewer needs Streamlit: pip install 'sparsehydro[viewer]'")
    sys.argv = build_argv(sys.argv[1:])
    sys.exit(stcli.main())


if __name__ == "__main__":  # pragma: no cover
    main()
