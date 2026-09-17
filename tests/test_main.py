"""Tests for the server entrypoint."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from src import main as main_module

RUNNER = CliRunner()


def _clear_logging_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the logging bootstrap variables from the environment."""
    for name in ("LOG_LEVEL", "LOG_STRUCTURED", "LOG_FILE"):
        monkeypatch.delenv(name, raising=False)


class TestLogParamsFromEnv:
    def test_defaults_to_info_plain_no_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without logging env vars, bootstrap defaults apply."""
        # Given
        _clear_logging_env(monkeypatch)
        # When
        level, structured, log_file = main_module._log_params_from_env()
        # Then
        assert level == logging.INFO
        assert structured is False
        assert log_file is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("debug", logging.DEBUG),
            ("WARNING", logging.WARNING),
            ("bogus", logging.INFO),
        ],
    )
    def test_parses_level_with_info_fallback(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
    ) -> None:
        """Level names parse case-insensitively; unknown names fall back."""
        # Given
        monkeypatch.setenv("LOG_LEVEL", raw)
        # When
        level, _, _ = main_module._log_params_from_env()
        # Then
        assert level == expected

    @pytest.mark.parametrize("raw", ["1", "true", "YES"])
    def test_parses_truthy_structured_values(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """Truthy LOG_STRUCTURED values enable JSON output."""
        # Given
        monkeypatch.setenv("LOG_STRUCTURED", raw)
        # When
        _, structured, _ = main_module._log_params_from_env()
        # Then
        assert structured is True

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_blank_level_falls_back_to_info(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """Empty or whitespace-only LOG_LEVEL means INFO."""
        # Given
        monkeypatch.setenv("LOG_LEVEL", raw)
        # When
        level, _, _ = main_module._log_params_from_env()
        # Then
        assert level == logging.INFO

    def test_blank_log_file_becomes_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whitespace-only LOG_FILE means no file sink."""
        # Given
        monkeypatch.setenv("LOG_FILE", "   ")
        # When
        _, _, log_file = main_module._log_params_from_env()
        # Then
        assert log_file is None

    def test_reads_log_file_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """LOG_FILE passes through trimmed."""
        # Given
        target = tmp_path / "rag.log"
        monkeypatch.setenv("LOG_FILE", str(target))
        # When
        _, _, log_file = main_module._log_params_from_env()
        # Then
        assert log_file == str(target)


def _patch_entrypoint(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the entrypoint's collaborators and record their calls."""
    recorded: dict[str, Any] = {}

    def fake_create_logger(name: str, **kwargs: Any) -> Any:
        recorded["logger"] = (name, kwargs)
        return SimpleNamespace(info=lambda *args, **kwargs: None)

    def fake_run(app: Any, **kwargs: Any) -> None:
        recorded["run"] = (app, kwargs)

    monkeypatch.setattr("src.create_logger", fake_create_logger)
    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr("src.api.app.create_app", lambda retriever: "APP")
    monkeypatch.setattr("src.app.adapter.LocalRetriever", lambda: "RETRIEVER")
    monkeypatch.setattr(
        "src.config.settings.app_settings",
        SimpleNamespace(HOST="127.0.0.1", PORT=9999),
    )
    return recorded


class TestMain:
    def test_configures_logging_then_serves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configure from env first, then run uvicorn with settings host/port."""
        # Given
        _clear_logging_env(monkeypatch)
        monkeypatch.setenv("LOG_LEVEL", "debug")
        recorded = _patch_entrypoint(monkeypatch)
        # When
        result = RUNNER.invoke(main_module.app, [])
        # Then
        assert result.exit_code == 0
        name, logger_kwargs = recorded["logger"]
        assert name == "src.main"
        assert logger_kwargs == {
            "level": logging.DEBUG,
            "structured": False,
            "log_file": None,
        }
        assert recorded["run"] == ("APP", {"host": "127.0.0.1", "port": 9999})

    def test_host_and_port_flags_override_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI flags win over settings, which .env outranks the shell env for."""
        # Given
        _clear_logging_env(monkeypatch)
        recorded = _patch_entrypoint(monkeypatch)
        # When
        result = RUNNER.invoke(main_module.app, ["--host", "1.2.3.4", "--port", "5001"])
        # Then
        assert result.exit_code == 0
        assert recorded["run"] == ("APP", {"host": "1.2.3.4", "port": 5001})
