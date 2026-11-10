"""Unit tests for midnight-rotating `logs/app.log` logging configuration."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest

from agentic_rag_notebook.config.logging_config import (
    _CONFIG_SIG_ATTR,
    _MANAGED_HANDLER_ATTR,
    LOGGER_NAME,
    configure_logging,
    get_logger,
)
from agentic_rag_notebook.config.settings import Settings


@pytest.fixture(autouse=True)
def _cleanup_managed_log_handlers() -> Iterator[None]:
    """Ensure tests do not leave file handlers attached to the shared package logger."""
    yield
    pkg_logger = logging.getLogger(LOGGER_NAME)
    for handler in list(pkg_logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTR, False):
            pkg_logger.removeHandler(handler)
            handler.close()
    if hasattr(pkg_logger, _CONFIG_SIG_ATTR):
        delattr(pkg_logger, _CONFIG_SIG_ATTR)


def _get_file_handler() -> TimedRotatingFileHandler:
    pkg_logger = logging.getLogger(LOGGER_NAME)
    for handler in pkg_logger.handlers:
        if isinstance(handler, TimedRotatingFileHandler) and getattr(
            handler, _MANAGED_HANDLER_ATTR, False
        ):
            return handler
    raise AssertionError("Expected a managed TimedRotatingFileHandler on package logger")


def test_configure_logging_creates_logs_folder_and_app_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        log_dir=log_dir,
        log_file_name="app.log",
        log_level="INFO",
        log_backup_count=7,
    )

    configure_logging(settings, console=False)

    log = get_logger("agentic_rag_notebook.test.app_log")
    log.info("Startup complete")
    handler = _get_file_handler()
    handler.flush()

    expected_file = log_dir / "app.log"
    assert log_dir.is_dir()
    assert expected_file.is_file()
    assert handler.when == "MIDNIGHT"
    assert handler.suffix == "%Y-%m-%d"
    assert handler.backupCount == 7

    content = expected_file.read_text(encoding="utf-8")
    assert "INFO" in content
    assert re.search(
        r"agentic_rag_notebook\.test\.app_log:test_configure_logging_creates_logs_folder_and_app_log:\d+",
        content,
    )
    assert "Startup complete" in content


def test_timed_rotating_handler_rolls_over_to_dated_archive_and_recreates_app_log(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(
        log_dir=log_dir,
        file_name="app.log",
        level="DEBUG",
        backup_count=3,
        console=False,
    )
    log = get_logger("agentic_rag_notebook.test.rollover")
    handler = _get_file_handler()

    log.debug("Day one debug entry")
    handler.flush()

    # Simulate midnight rollover for 2026-11-10 -> creates app.log.2026-11-10 and new app.log
    target_ts = int(time.mktime((2026, 11, 10, 0, 0, 0, 0, 0, -1)))
    handler.rolloverAt = target_ts + handler.interval
    handler.doRollover()

    log.info("Day two info entry")
    handler.flush()

    archived = log_dir / "app.log.2026-11-10"
    active = log_dir / "app.log"
    assert archived.is_file()
    assert active.is_file()
    assert "Day one debug entry" in archived.read_text(encoding="utf-8")
    assert "Day two info entry" in active.read_text(encoding="utf-8")
    assert "Day two info entry" not in archived.read_text(encoding="utf-8")


def test_timed_rotating_handler_prunes_archives_beyond_backup_count(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(
        log_dir=log_dir,
        file_name="app.log",
        level="INFO",
        backup_count=2,
        console=False,
    )
    log = get_logger("agentic_rag_notebook.test.retention")
    handler = _get_file_handler()

    for day_num in (1, 2, 3, 4):
        log.info("Entry for day %d", day_num)
        handler.flush()
        day_ts = int(time.mktime((2026, 11, day_num, 0, 0, 0, 0, 0, -1)))
        handler.rolloverAt = day_ts + handler.interval
        handler.doRollover()

    archives = sorted(p.name for p in log_dir.glob("app.log.*"))
    assert archives == ["app.log.2026-11-03", "app.log.2026-11-04"]
    assert (log_dir / "app.log").is_file()


def test_configure_logging_is_idempotent_across_reruns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log_dir = tmp_path / "logs"

    configure_logging(
        log_dir=log_dir,
        file_name="app.log",
        level="INFO",
        backup_count=5,
        console=False,
    )
    configure_logging(
        log_dir=log_dir,
        file_name="app.log",
        level="INFO",
        backup_count=5,
        console=False,
    )

    log = get_logger("agentic_rag_notebook.test.idempotent")
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log.info("Single message")

    _get_file_handler().flush()

    lines = (log_dir / "app.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "Single message" in lines[0]
    assert caplog.messages == ["Single message"]
