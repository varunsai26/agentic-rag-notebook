"""Centralized midnight-rotating logging configuration (`<log_dir>/app.log` -> `app.log.YYYY-MM-DD`)."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag_notebook.config.settings import Settings

LOGGER_NAME = "agentic_rag_notebook"
DEFAULT_LOG_FILENAME = "app.log"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MANAGED_HANDLER_ATTR = "_agentic_rag_managed"
_CONFIG_SIG_ATTR = "_agentic_rag_config_sig"


def _resolve_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    mapping = logging.getLevelNamesMapping()
    if normalized not in mapping:
        raise ValueError(f"Unsupported log level: {level!r}")
    return mapping[normalized]


def configure_logging(
    settings: Settings | None = None,
    *,
    log_dir: Path | None = None,
    file_name: str | None = None,
    level: str | int | None = None,
    backup_count: int | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure package logging using TimedRotatingFileHandler (`<log_dir>/app.log`).

    Rotates automatically at midnight into `<log_dir>/app.log.YYYY-MM-DD` and creates a fresh
    `<log_dir>/app.log` for every new date. Idempotent across Streamlit reruns while keeping
    `propagate = True` so pytest `caplog` continues to capture package logs.
    """
    if settings is not None:
        target_dir = Path(log_dir) if log_dir is not None else settings.log_dir
        target_filename = file_name if file_name is not None else settings.log_file_name
        target_level: str | int = level if level is not None else settings.log_level
        target_backup = backup_count if backup_count is not None else settings.log_backup_count
    elif log_dir is None or file_name is None or level is None or backup_count is None:
        from agentic_rag_notebook.config.settings import get_settings

        active_settings = get_settings()
        target_dir = Path(log_dir) if log_dir is not None else active_settings.log_dir
        target_filename = file_name if file_name is not None else active_settings.log_file_name
        target_level = level if level is not None else active_settings.log_level
        target_backup = (
            backup_count if backup_count is not None else active_settings.log_backup_count
        )
    else:
        target_dir = Path(log_dir)
        target_filename = file_name
        target_level = level
        target_backup = backup_count

    clean_filename = Path(target_filename).name or DEFAULT_LOG_FILENAME
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / clean_filename
    numeric_level = _resolve_log_level(target_level)
    safe_backup_count = max(0, target_backup)

    pkg_logger = logging.getLogger(LOGGER_NAME)
    signature = (os.fspath(log_path.resolve()), numeric_level, safe_backup_count, console)
    managed_handlers = [h for h in pkg_logger.handlers if getattr(h, _MANAGED_HANDLER_ATTR, False)]
    if (
        managed_handlers
        and getattr(pkg_logger, _CONFIG_SIG_ATTR, None) == signature
        and log_path.exists()
    ):
        return pkg_logger

    for existing in managed_handlers:
        pkg_logger.removeHandler(existing)
        existing.close()

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    file_handler = TimedRotatingFileHandler(
        filename=os.fspath(log_path),
        when="midnight",
        interval=1,
        backupCount=safe_backup_count,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_HANDLER_ATTR, True)
    pkg_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setLevel(numeric_level)
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _MANAGED_HANDLER_ATTR, True)
        pkg_logger.addHandler(stream_handler)

    pkg_logger.setLevel(numeric_level)
    pkg_logger.propagate = True
    setattr(pkg_logger, _CONFIG_SIG_ATTR, signature)
    return pkg_logger


def get_logger(name: str) -> logging.Logger:
    """Return a module logger scoped under the application logger hierarchy."""
    return logging.getLogger(name)
