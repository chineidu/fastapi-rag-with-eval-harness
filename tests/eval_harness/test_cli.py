"""Tests for the eval_harness.cli module."""

import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from src.eval_harness.cli import _category_means, app, load_adapter
from src.eval_harness.loader import GroundTruthLoader
from src.eval_harness.runner import EvalRunner
from src.eval_harness.store import ResultStore
from src.schemas.harness import RetrievalResult, RetrievedDocument
from src.schemas.models import GroundTruthRecord

RUNNER = CliRunner()


class StubAdapter:
    """Deterministic adapter returning a single configured document."""

    def __init__(self, doc: str = "docs/quickstart.md") -> None:
        """Initialize with the document to return."""
        self._doc = doc

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Return the configured document with a fixed score."""
        return RetrievalResult(documents=[RetrievedDocument(self._doc, 0.9)])

    def generate(self, query: str, documents: list[str]) -> str:
        """Return a stub answer."""
        return "stub answer"


def _write_ground_truth(tmp_path: Path, records: list[dict]) -> Path:
    """Write records as JSONL and return the file path."""
    path = tmp_path / "ground_truth.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _answerable_record(query_id: str = "q1") -> dict:
    """Build a minimal answerable ground truth record."""
    return {
        "query_id": query_id,
        "label": "DIRECT_LOOKUP",
        "query_text": "How do I set a default query parameter?",
        "relevant_docs": ["docs/quickstart.md"],
    }


def _unanswerable_record(query_id: str = "q2") -> dict:
    """Build a minimal unanswerable ground truth record."""
    return {
        "query_id": query_id,
        "label": "CONCEPTUAL",
        "query_text": "Why is this failing?",
        "relevant_docs": [],
        "answerable": False,
    }


def _fake_adapter_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str = "fake_adapter",
) -> types.ModuleType:
    """Register a fake adapter module in sys.modules and return it."""
    module: Any = types.ModuleType(name)
    module.FakeAdapter = StubAdapter
    module.nested = types.SimpleNamespace(Deep=StubAdapter)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _seed_run(
    tmp_path: Path,
    records: Sequence[GroundTruthRecord],
    tag: str,
    doc: str = "docs/quickstart.md",
) -> int:
    """Run EvalRunner directly against the shared test db and return the run id."""
    db = tmp_path / "runs.db"
    with ResultStore(db) as store:
        runner = EvalRunner(
            StubAdapter(doc),
            store,
            records,
            k=1,
            concurrency=1,
            max_retries=0,
        )
        summary = runner.run(tag)
    return summary.run_id


class TestLoadAdapter:
    """Tests for the load_adapter import-path resolver."""

    def test_module_colon_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a module:Class path, then the adapter is instantiated."""
        # Given
        _fake_adapter_module(monkeypatch)

        # When
        adapter = load_adapter("fake_adapter:FakeAdapter")

        # Then
        assert adapter.retrieve("q", 1).documents[0].doc_path == "docs/quickstart.md"

    def test_module_attr_without_colon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a module.Class path, then the adapter is instantiated."""
        # Given
        _fake_adapter_module(monkeypatch)

        # When
        adapter = load_adapter("fake_adapter.FakeAdapter")

        # Then
        assert adapter.retrieve("q", 1).documents[0].doc_path == "docs/quickstart.md"

    def test_dotted_nested_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a dotted attribute path, then it is resolved."""
        # Given
        _fake_adapter_module(monkeypatch)

        # When
        adapter = load_adapter("fake_adapter:nested.Deep")

        # Then
        assert adapter.retrieve("q", 1).documents[0].doc_path == "docs/quickstart.md"

    def test_missing_module_raises(self) -> None:
        """Given an unknown module, then ImportError is raised."""
        # Given / When / Then
        with pytest.raises(ImportError, match="Could not import module"):
            load_adapter("no_such_module_xyz:FakeAdapter")

    def test_missing_attribute_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a missing attribute, then ImportError is raised."""
        # Given
        _fake_adapter_module(monkeypatch)

        # When / Then
        with pytest.raises(ImportError, match="not found"):
            load_adapter("fake_adapter:Missing")

    def test_non_callable_target_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a non-callable target, then TypeError is raised."""
        # Given
        module: Any = types.ModuleType("fake_mod_config")
        module.Config = object()
        monkeypatch.setitem(sys.modules, "fake_mod_config", module)

        # When / Then
        with pytest.raises(TypeError, match="not callable"):
            load_adapter("fake_mod_config:Config")

    def test_missing_retrieve_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given an instance without retrieve(), then TypeError is raised."""
        # Given
        module: Any = types.ModuleType("fake_mod_nor")
        module.NoRetrieve = type("NoRetrieve", (), {})
        monkeypatch.setitem(sys.modules, "fake_mod_nor", module)

        # When / Then
        with pytest.raises(TypeError, match="retrieve"):
            load_adapter("fake_mod_nor:NoRetrieve")

    def test_empty_path_raises(self) -> None:
        """Given an empty import path, then ValueError is raised."""
        # Given / When / Then
        with pytest.raises(ValueError, match="empty"):
            load_adapter("")


class TestRunCommand:
    """Tests for the rag-eval run command."""

    def test_run_complete_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a valid run, then exit code is 0 and results are persisted."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        monkeypatch.setattr(
            "src.eval_harness.cli.load_adapter", lambda path: StubAdapter()
        )
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--adapter",
                "stub:x",
                "--ground-truth",
                str(gt),
                "--db",
                str(db),
                "--tag",
                "v1-baseline",
                "--k",
                "5",
                "--concurrency",
                "1",
            ],
        )

        # Then
        assert result.exit_code == 0
        assert "v1-baseline" in result.output
        with ResultStore(db) as store:
            runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["tag"] == "v1-baseline"
        assert runs[0]["status"] == "complete"

    def test_run_partial_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a failing adapter, then exit code is 1 with partial status."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])

        class FailingAdapter:
            """Adapter that always raises."""

            def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
                raise ValueError("boom")

        monkeypatch.setattr(
            "src.eval_harness.cli.load_adapter", lambda path: FailingAdapter()
        )
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--adapter",
                "stub:x",
                "--ground-truth",
                str(gt),
                "--db",
                str(db),
                "--tag",
                "t",
                "--concurrency",
                "1",
            ],
        )

        # Then
        assert result.exit_code == 1
        assert "partial" in result.output

    def test_run_auto_generates_tag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given no --tag, then an auto- tag is generated."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        monkeypatch.setattr(
            "src.eval_harness.cli.load_adapter", lambda path: StubAdapter()
        )
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--adapter",
                "stub:x",
                "--ground-truth",
                str(gt),
                "--db",
                str(db),
            ],
        )

        # Then
        assert result.exit_code == 0
        assert "auto-" in result.output
        with ResultStore(db) as store:
            runs = store.list_runs()
        assert runs[0]["tag"].startswith("auto-")

    def test_run_missing_adapter_exits_two(self, tmp_path: Path) -> None:
        """Given no adapter configured, then exit code is 2."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        cfg = tmp_path / ".rag-eval.yaml"
        cfg.write_text('adapter: ""\n', encoding="utf-8")
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--config",
                str(cfg),
                "--ground-truth",
                str(gt),
                "--db",
                str(db),
            ],
        )

        # Then
        assert result.exit_code == 2
        assert "no adapter" in result.output

    def test_run_missing_ground_truth_exits_two(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a missing ground truth file, then exit code is 2."""
        # Given
        monkeypatch.setattr(
            "src.eval_harness.cli.load_adapter", lambda path: StubAdapter()
        )
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--adapter",
                "stub:x",
                "--ground-truth",
                str(tmp_path / "nope.jsonl"),
                "--db",
                str(db),
            ],
        )

        # Then
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_run_zero_k_exits_two(self, tmp_path: Path) -> None:
        """Given --k 0, then exit code is 2 with a clean error."""
        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--adapter",
                "stub:x",
                "--ground-truth",
                str(tmp_path / "gt.jsonl"),
                "--db",
                str(tmp_path / "runs.db"),
                "--k",
                "0",
            ],
        )

        # Then
        assert result.exit_code == 2
        assert "k must be positive" in result.output

    def test_run_invalid_adapter_path_exits_two(self, tmp_path: Path) -> None:
        """Given an unimportable adapter path, then exit code is 2."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(
            app,
            [
                "run",
                "--adapter",
                "no_such_module_x:Foo",
                "--ground-truth",
                str(gt),
                "--db",
                str(db),
            ],
        )

        # Then
        assert result.exit_code == 2
        assert "Could not import" in result.output


class TestDiffCommand:
    """Tests for the rag-eval diff command."""

    def _invoke(self, tmp_path: Path, args: list[str]):
        """Invoke diff against the shared test db."""
        return RUNNER.invoke(app, ["diff", *args, "--db", str(tmp_path / "runs.db")])

    def test_regression_exits_one(self, tmp_path: Path) -> None:
        """Given a regression, then exit code is 1 and the row is flagged."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        records = GroundTruthLoader.load(gt)
        _seed_run(tmp_path, records, "baseline", doc="docs/quickstart.md")
        _seed_run(tmp_path, records, "current", doc="docs/other.md")

        # When
        result = self._invoke(tmp_path, ["baseline", "current"])

        # Then
        assert result.exit_code == 1
        assert "regressed" in result.output
        assert "0.000" in result.output

    def test_no_change_exits_zero(self, tmp_path: Path) -> None:
        """Given identical runs, then exit code is 0 with no flags."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        records = GroundTruthLoader.load(gt)
        _seed_run(tmp_path, records, "baseline", doc="docs/quickstart.md")
        _seed_run(tmp_path, records, "current", doc="docs/quickstart.md")

        # When
        result = self._invoke(tmp_path, ["baseline", "current"])

        # Then
        assert result.exit_code == 0
        assert "1.000" in result.output

    def test_missing_run_exits_two(self, tmp_path: Path) -> None:
        """Given an unknown run reference, then exit code is 2."""
        # When
        result = self._invoke(tmp_path, ["nope", "also-nope"])

        # Then
        assert result.exit_code == 2
        assert "No run found" in result.output

    def test_negative_threshold_exits_two(self, tmp_path: Path) -> None:
        """Given a negative threshold, then exit code is 2 with a clean error."""
        # When
        result = self._invoke(
            tmp_path, ["baseline", "current", "--threshold-absolute", "-1"]
        )

        # Then
        assert result.exit_code == 2
        assert "threshold_absolute must not be negative" in result.output

    def test_category_means_tolerates_null_ground_truth(self) -> None:
        """Given a row with null ground_truth, then it is excluded without crashing."""
        # Given
        rows = [
            {
                "query_id": "q1",
                "category": "DIRECT_LOOKUP",
                "k_value": 1,
                "recall_at_k": 0.5,
                "ground_truth": None,
            }
        ]

        # When
        means, excluded = _category_means(rows)

        # Then
        assert means == {}
        assert excluded == 1

    def test_excludes_unanswerable_from_means(self, tmp_path: Path) -> None:
        """Given unanswerable rows, then they are excluded from category means."""
        # Given
        gt = _write_ground_truth(
            tmp_path,
            [_answerable_record(), _unanswerable_record()],
        )
        records = GroundTruthLoader.load(gt)
        _seed_run(tmp_path, records, "baseline", doc="docs/quickstart.md")
        _seed_run(tmp_path, records, "current", doc="docs/quickstart.md")

        # When
        result = self._invoke(tmp_path, ["baseline", "current"])

        # Then
        assert result.exit_code == 0
        overall = next(
            line for line in result.output.splitlines() if line.startswith("OVERALL")
        )
        assert "1.000" in overall
        assert "0.500" not in overall

    def test_all_unanswerable_reports_no_comparison(self, tmp_path: Path) -> None:
        """Given runs with only unanswerable rows, then diff exits zero with a note."""
        # Given
        gt = _write_ground_truth(
            tmp_path,
            [_unanswerable_record("q1"), _unanswerable_record("q2")],
        )
        records = GroundTruthLoader.load(gt)
        _seed_run(tmp_path, records, "baseline", doc="docs/quickstart.md")
        _seed_run(tmp_path, records, "current", doc="docs/quickstart.md")

        # When
        result = self._invoke(tmp_path, ["baseline", "current"])

        # Then
        assert result.exit_code == 0
        assert "No answerable queries" in result.output

    def test_verbose_shows_per_query_deltas(self, tmp_path: Path) -> None:
        """Given --verbose, then per-query delta rows are printed."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        records = GroundTruthLoader.load(gt)
        _seed_run(tmp_path, records, "baseline", doc="docs/quickstart.md")
        _seed_run(tmp_path, records, "current", doc="docs/other.md")

        # When
        result = self._invoke(tmp_path, ["baseline", "current", "--verbose"])

        # Then
        assert result.exit_code == 1
        assert "Per-query" in result.output
        assert "q1" in result.output


class TestListCommand:
    """Tests for the rag-eval list command."""

    def test_empty_db(self, tmp_path: Path) -> None:
        """Given an empty database, then a friendly message is printed."""
        # When
        result = RUNNER.invoke(app, ["list", "--db", str(tmp_path / "empty.db")])

        # Then
        assert result.exit_code == 0
        assert "No runs yet" in result.output

    def test_lists_runs(self, tmp_path: Path) -> None:
        """Given seeded runs, then they appear in the list output."""
        # Given
        gt = _write_ground_truth(tmp_path, [_answerable_record()])
        records = GroundTruthLoader.load(gt)
        _seed_run(tmp_path, records, "v1-baseline")
        db = tmp_path / "runs.db"

        # When
        result = RUNNER.invoke(app, ["list", "--db", str(db)])

        # Then
        assert result.exit_code == 0
        assert "v1-baseline" in result.output
        assert "QUERIES" in result.output
        assert "1/1" in result.output
