"""CLI entry point to launch the Agentic RAG Notebook Streamlit app."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from agentic_rag_notebook.config.logging_config import configure_logging
from agentic_rag_notebook.config.settings import get_settings

logger = logging.getLogger("agentic_rag_notebook.ui.runner")


def main() -> None:
    """Launch Streamlit targeting ui/app.py."""
    settings = get_settings()
    configure_logging(settings)
    app_path = Path(__file__).resolve().parent / "app.py"
    logger.info("Launching Streamlit UI app=%s log_dir=%s", app_path, settings.log_dir)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
