import logging
import logging.handlers

import pytest

from src import EmojiFormatter, _build_formatter, create_logger


class TestEmojiFormatter:
    def test_adds_debug_emoji(self) -> None:
        fmt = EmojiFormatter(fmt="%(emoji)s %(message)s")
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "msg", (), None)
        assert fmt.format(record).startswith("🐛")

    def test_adds_info_emoji(self) -> None:
        fmt = EmojiFormatter(fmt="%(emoji)s %(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert fmt.format(record).startswith("ℹ️")

    def test_adds_warning_emoji(self) -> None:
        fmt = EmojiFormatter(fmt="%(emoji)s %(message)s")
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert fmt.format(record).startswith("⚠️")

    def test_adds_error_emoji(self) -> None:
        fmt = EmojiFormatter(fmt="%(emoji)s %(message)s")
        record = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        assert fmt.format(record).startswith("❌")

    def test_adds_critical_emoji(self) -> None:
        fmt = EmojiFormatter(fmt="%(emoji)s %(message)s")
        record = logging.LogRecord("test", logging.CRITICAL, "", 0, "msg", (), None)
        assert fmt.format(record).startswith("🚨")

    def test_unknown_level_has_empty_emoji(self) -> None:
        fmt = EmojiFormatter(fmt="%(emoji)s %(message)s")
        record = logging.LogRecord("test", 99, "", 0, "msg", (), None)
        assert fmt.format(record).startswith(" ")


class TestBuildFormatter:
    def test_structured_returns_json_formatter(self) -> None:
        fmt = _build_formatter(structured=True)
        from pythonjsonlogger import json as jsonlogger

        assert isinstance(fmt, jsonlogger.JsonFormatter)

    def test_plain_returns_emoji_formatter(self) -> None:
        fmt = _build_formatter(structured=False)
        assert isinstance(fmt, EmojiFormatter)


class TestCreateLogger:
    def test_returns_logger_with_queue_handler(self) -> None:
        logger = create_logger("test_logger")
        handlers = logger.handlers
        assert any(isinstance(h, logging.handlers.QueueHandler) for h in handlers)

    def test_propagate_is_false(self) -> None:
        logger = create_logger("test_logger_2")
        assert logger.propagate is False

    def test_same_name_returns_same_logger(self) -> None:
        a = create_logger("unique_test_name")
        b = create_logger("unique_test_name")
        assert a is b

    def test_no_duplicate_queue_handlers(self) -> None:
        logger = create_logger("no_dup_test")
        initial = sum(
            1 for h in logger.handlers if isinstance(h, logging.handlers.QueueHandler)
        )
        create_logger("no_dup_test")
        after = sum(
            1 for h in logger.handlers if isinstance(h, logging.handlers.QueueHandler)
        )
        assert after == initial

    def test_warns_when_reinitialized_with_different_structured(self) -> None:
        with pytest.warns(UserWarning, match="structured"):
            create_logger("warn_test", structured=True, level=logging.DEBUG)

    def test_warns_when_reinitialized_with_different_level(self) -> None:
        with pytest.warns(UserWarning, match="level"):
            create_logger("warn_test_2", level=logging.DEBUG)

    def test_warns_when_reinitialized_with_different_log_file(self) -> None:
        with pytest.warns(UserWarning, match="log_file"):
            create_logger("warn_test_3", log_file="/tmp/fake.log")
