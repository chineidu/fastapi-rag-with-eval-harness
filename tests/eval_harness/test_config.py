"""Tests for the eval_harness.config module."""

import pytest

from src.eval_harness.config import (
    _from_dict,
    apply_cli_overrides,
    load_harness_config,
)
from src.schemas.containers import (
    HarnessConfig,
    HarnessDefaults,
    HarnessDiffThresholds,
)


class TestHarnessDefaults:
    """Tests for HarnessDefaults dataclass defaults."""

    def test_default_k(self) -> None:
        """Given no arguments, then k defaults to 10."""
        # Given / When
        d = HarnessDefaults()

        # Then
        assert d.k == 10

    def test_default_concurrency(self) -> None:
        """Given no arguments, then concurrency defaults to 3."""
        # Given / When
        d = HarnessDefaults()

        # Then
        assert d.concurrency == 3

    def test_custom_values(self) -> None:
        """Given explicit values, then they override defaults."""
        # Given / When
        d = HarnessDefaults(k=5, concurrency=8)

        # Then
        assert d.k == 5
        assert d.concurrency == 8


class TestHarnessDiffThresholds:
    """Tests for HarnessDiffThresholds dataclass defaults."""

    def test_defaults(self) -> None:
        """Given no arguments, then thresholds are 0.05 and 5.0."""
        # Given / When
        t = HarnessDiffThresholds()

        # Then
        assert t.threshold_absolute == 0.05
        assert t.threshold_relative == 5.0

    def test_custom_values(self) -> None:
        """Given explicit values, then they override defaults."""
        # Given / When
        t = HarnessDiffThresholds(threshold_absolute=0.1, threshold_relative=10.0)

        # Then
        assert t.threshold_absolute == 0.1
        assert t.threshold_relative == 10.0


class TestHarnessConfig:
    """Tests for HarnessConfig dataclass defaults."""

    def test_defaults(self) -> None:
        """Given no arguments, then config uses sensible defaults."""
        # Given / When
        cfg = HarnessConfig()

        # Then
        assert cfg.adapter == ""
        assert cfg.ground_truth == "data/ground_truth.jsonl"
        assert cfg.db == "data/.rag-eval/runs.db"
        assert isinstance(cfg.defaults, HarnessDefaults)
        assert isinstance(cfg.diff, HarnessDiffThresholds)

    def test_custom_values(self) -> None:
        """Given explicit values, then they override defaults."""
        # Given
        defaults = HarnessDefaults(k=20, concurrency=5)
        diff = HarnessDiffThresholds(threshold_absolute=0.1)

        # When
        cfg = HarnessConfig(
            adapter="tests.stub:StubAdapter",
            ground_truth="data/test.json",
            db="data/test.db",
            defaults=defaults,
            diff=diff,
        )

        # Then
        assert cfg.adapter == "tests.stub:StubAdapter"
        assert cfg.ground_truth == "data/test.json"
        assert cfg.db == "data/test.db"
        assert cfg.defaults.k == 20
        assert cfg.diff.threshold_absolute == 0.1


class TestLoadHarnessConfig:
    """Tests for load_harness_config function."""

    def test_loads_defaults_when_no_file(self, tmp_path) -> None:
        """Given a non-existent path, then defaults are returned."""
        # Given
        nonexistent = tmp_path / "nonexistent.yaml"

        # When
        cfg = load_harness_config(nonexistent)

        # Then
        assert cfg.adapter == ""
        assert cfg.defaults.k == 10

    def test_loads_from_yaml_file(self, tmp_path) -> None:
        """Given a valid YAML file, then config is loaded from it."""
        # Given
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "adapter: my.adapter\ndefaults:\n  k: 20\n  concurrency: 5\n"
        )

        # When
        cfg = load_harness_config(config_file)

        # Then
        assert cfg.adapter == "my.adapter"
        assert cfg.defaults.k == 20
        assert cfg.defaults.concurrency == 5

    def test_loads_from_current_directory(self, tmp_path, monkeypatch) -> None:
        """Given .rag-eval.yaml in cwd, then it is loaded automatically."""
        # Given
        config_file = tmp_path / ".rag-eval.yaml"
        config_file.write_text("adapter: auto-loaded\n")
        monkeypatch.chdir(tmp_path)

        # When
        cfg = load_harness_config()

        # Then
        assert cfg.adapter == "auto-loaded"

    def test_invalid_yaml_raises(self, tmp_path) -> None:
        """Given invalid YAML content, then a TypeError or yaml error is raised."""
        # Given
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(": : invalid yaml {{{")

        # When / Then
        with pytest.raises((TypeError, Exception)):
            load_harness_config(config_file)


class TestApplyCliOverrides:
    """Tests for apply_cli_overrides function."""

    def test_no_overrides_preserves_config(self) -> None:
        """Given no overrides, then config is unchanged."""
        # Given
        cfg = HarnessConfig(adapter="test", defaults=HarnessDefaults(k=20))

        # When
        result = apply_cli_overrides(cfg)

        # Then
        assert result.adapter == "test"
        assert result.defaults.k == 20

    def test_adapter_override(self) -> None:
        """Given adapter override, then it replaces the config value."""
        # Given
        cfg = HarnessConfig(adapter="old")

        # When
        result = apply_cli_overrides(cfg, adapter="new")

        # Then
        assert result.adapter == "new"

    def test_k_override(self) -> None:
        """Given k override, then it replaces the config value."""
        # Given
        cfg = HarnessConfig(defaults=HarnessDefaults(k=10))

        # When
        result = apply_cli_overrides(cfg, k=25)

        # Then
        assert result.defaults.k == 25

    def test_invalid_k_override_raises(self) -> None:
        """Given k=0 override, then ValueError is raised."""
        # Given
        cfg = HarnessConfig()

        # When / Then
        with pytest.raises(ValueError, match="k must be positive"):
            apply_cli_overrides(cfg, k=0)

    def test_negative_threshold_override_raises(self) -> None:
        """Given negative threshold override, then ValueError is raised."""
        # Given
        cfg = HarnessConfig()

        # When / Then
        with pytest.raises(ValueError, match="threshold_absolute must not be negative"):
            apply_cli_overrides(cfg, threshold_absolute=-0.1)

    def test_concurrency_override(self) -> None:
        """Given concurrency override, then it replaces the config value."""
        # Given
        cfg = HarnessConfig(defaults=HarnessDefaults(concurrency=3))

        # When
        result = apply_cli_overrides(cfg, concurrency=8)

        # Then
        assert result.defaults.concurrency == 8

    def test_threshold_override(self) -> None:
        """Given threshold overrides, then they replace the config values."""
        # Given
        cfg = HarnessConfig(diff=HarnessDiffThresholds())

        # When
        result = apply_cli_overrides(
            cfg, threshold_absolute=0.2, threshold_relative=10.0
        )

        # Then
        assert result.diff.threshold_absolute == 0.2
        assert result.diff.threshold_relative == 10.0

    def test_ground_truth_override(self) -> None:
        """Given ground_truth override, then it replaces the config value."""
        # Given
        cfg = HarnessConfig(ground_truth="old.json")

        # When
        result = apply_cli_overrides(cfg, ground_truth="new.json")

        # Then
        assert result.ground_truth == "new.json"

    def test_db_override(self) -> None:
        """Given db override, then it replaces the config value."""
        # Given
        cfg = HarnessConfig(db="old.db")

        # When
        result = apply_cli_overrides(cfg, db="new.db")

        # Then
        assert result.db == "new.db"


class TestFromDict:
    """Tests for _from_dict validation branches."""

    def test_rejects_zero_k(self) -> None:
        """Given k=0, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="k must be positive"):
            _from_dict({"defaults": {"k": 0}})

    def test_rejects_negative_k(self) -> None:
        """Given negative k, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="k must be positive"):
            _from_dict({"defaults": {"k": -1}})

    def test_rejects_zero_concurrency(self) -> None:
        """Given concurrency=0, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="concurrency must be positive"):
            _from_dict({"defaults": {"concurrency": 0}})

    def test_rejects_negative_concurrency(self) -> None:
        """Given negative concurrency, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="concurrency must be positive"):
            _from_dict({"defaults": {"concurrency": -5}})

    def test_rejects_negative_threshold_absolute(self) -> None:
        """Given negative threshold_absolute, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="threshold_absolute must not be negative"):
            _from_dict({"diff": {"threshold_absolute": -0.1}})

    def test_rejects_negative_threshold_relative(self) -> None:
        """Given negative threshold_relative, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="threshold_relative must not be negative"):
            _from_dict({"diff": {"threshold_relative": -1.0}})

    def test_valid_config_passes(self) -> None:
        """Given valid config dict, then HarnessConfig is returned."""
        # Given
        data = {"adapter": "test", "defaults": {"k": 10, "concurrency": 3}}

        # When
        cfg = _from_dict(data)

        # Then
        assert cfg.adapter == "test"
        assert cfg.defaults.k == 10
