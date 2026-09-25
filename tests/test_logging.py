"""Logging setup tests."""

from collections.abc import Iterator

import pytest
import structlog

from jarvis.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def test_logger_created_before_configure_honours_level(capsys: pytest.CaptureFixture[str]) -> None:
    log = get_logger("early.module")  # like a module-level logger at import time
    configure_logging("WARNING")

    log.info("hidden_event")
    log.warning("shown_event")

    err = capsys.readouterr().err
    assert "shown_event" in err
    assert "early.module" in err
    assert "hidden_event" not in err
