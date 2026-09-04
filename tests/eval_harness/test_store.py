"""Tests for the eval_harness.store module."""

import sqlite3

import pytest

from src.eval_harness.store import ResultStore


class TestResultStoreInit:
    """Tests for ResultStore initialization."""

    def test_creates_database(self, tmp_path) -> None:
        """Given a path, then a SQLite database is created."""
        # Given
        db_path = tmp_path / "test.db"

        # When
        store = ResultStore(db_path)
        store.close()

        # Then
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path) -> None:
        """Given a nested path, then parent directories are created."""
        # Given
        db_path = tmp_path / "subdir" / "deep" / "test.db"

        # When
        store = ResultStore(db_path)
        store.close()

        # Then
        assert db_path.exists()


class TestResultStoreContextManager:
    """Tests for ResultStore context manager protocol."""

    def test_enter_returns_self(self, tmp_path) -> None:
        """Given a store, then __enter__ returns the store itself."""
        # Given
        store = ResultStore(tmp_path / "test.db")

        # When
        with store as ctx:
            # Then
            assert ctx is store

    def test_exit_closes_connection(self, tmp_path) -> None:
        """Given a store in a with block, then connection is closed on exit."""
        # Given / When
        with ResultStore(tmp_path / "test.db") as store:
            # Verify connection is open by running a query
            store._conn.execute("SELECT 1")

        # Then - connection should be closed, further operations should fail
        with pytest.raises(sqlite3.ProgrammingError):
            store._conn.execute("SELECT 1")


class TestResultStoreCreateRun:
    """Tests for ResultStore.create_run method."""

    def test_creates_run_with_tag(self, tmp_path) -> None:
        """Given a tag, then a run is created with that tag."""
        # Given
        store = ResultStore(tmp_path / "test.db")

        # When
        run_id = store.create_run("test-run")

        # Then
        assert isinstance(run_id, int)
        run = store.get_run(run_id)
        assert run is not None
        assert run["tag"] == "test-run"
        assert run["status"] == "running"
        store.close()

    def test_creates_run_with_config(self, tmp_path) -> None:
        """Given config dict, then it is stored as JSON."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        config = {"k": 10, "adapter": "stub"}

        # When
        run_id = store.create_run("test-run", config=config)

        # Then
        run = store.get_run(run_id)
        assert run is not None
        assert run["config"] is not None
        store.close()

    def test_creates_run_with_metadata(self, tmp_path) -> None:
        """Given metadata dict, then it is stored as JSON."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        metadata = {"note": "testing"}

        # When
        run_id = store.create_run("test-run", metadata=metadata)

        # Then
        run = store.get_run(run_id)
        assert run is not None
        assert run["metadata"] is not None
        store.close()

    def test_increments_run_ids(self, tmp_path) -> None:
        """Given multiple runs, then IDs are monotonically increasing."""
        # Given
        store = ResultStore(tmp_path / "test.db")

        # When
        id1 = store.create_run("run-1")
        id2 = store.create_run("run-2")

        # Then
        assert id2 > id1
        store.close()


class TestResultStoreSaveResult:
    """Tests for ResultStore.save_result method."""

    def test_saves_result(self, tmp_path) -> None:
        """Given a result, then it is persisted and retrievable."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        store.save_result(
            run_id=run_id,
            query_id="q1",
            category="DIRECT_LOOKUP",
            k_value=10,
            retrieved_docs=[["doc/a.md", 0.9]],
            ground_truth=["doc/a.md"],
            recall_at_k=1.0,
            precision_at_k=0.5,
            latency_ms=12.5,
        )

        # Then
        results = store.get_results(run_id)
        assert len(results) == 1
        assert results[0]["query_id"] == "q1"
        assert results[0]["recall_at_k"] == 1.0
        store.close()

    def test_saves_failed_result(self, tmp_path) -> None:
        """Given a failed result, then error is stored."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        store.save_result(
            run_id=run_id,
            query_id="q1",
            category="A",
            k_value=10,
            retrieved_docs=[],
            ground_truth=["doc.md"],
            recall_at_k=None,
            precision_at_k=None,
            latency_ms=50.0,
            status="fatal",
            error="Connection refused",
        )

        # Then
        results = store.get_results(run_id)
        assert results[0]["status"] == "fatal"
        assert results[0]["error"] == "Connection refused"
        store.close()

    def test_saves_with_metrics_extra(self, tmp_path) -> None:
        """Given metrics_extra dict, then it is stored as JSON."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        store.save_result(
            run_id=run_id,
            query_id="q1",
            category="A",
            k_value=5,
            retrieved_docs=[],
            ground_truth=[],
            recall_at_k=0.0,
            precision_at_k=0.0,
            latency_ms=1.0,
            metrics_extra={"custom": "value"},
        )

        # Then
        results = store.get_results(run_id)
        assert results[0]["metrics_extra"] is not None
        store.close()


class TestResultStoreFinalizeRun:
    """Tests for ResultStore.finalize_run method."""

    def test_finalize_as_complete(self, tmp_path) -> None:
        """Given status='complete', then run status is updated."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        store.finalize_run(run_id, "complete")

        # Then
        run = store.get_run(run_id)
        assert run is not None
        assert run["status"] == "complete"
        store.close()

    def test_finalize_as_partial(self, tmp_path) -> None:
        """Given status='partial', then run status is updated."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        store.finalize_run(run_id, "partial")

        # Then
        run = store.get_run(run_id)
        assert run is not None
        assert run["status"] == "partial"
        store.close()


class TestResultStoreGetRun:
    """Tests for ResultStore.get_run method."""

    def test_returns_none_for_missing(self, tmp_path) -> None:
        """Given a non-existent run_id, then None is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")

        # When
        result = store.get_run(999)

        # Then
        assert result is None
        store.close()

    def test_returns_run_dict(self, tmp_path) -> None:
        """Given an existing run, then a dict is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        result = store.get_run(run_id)

        # Then
        assert result is not None
        assert result["id"] == run_id
        assert result["tag"] == "test-run"
        store.close()


class TestResultStoreResolveRef:
    """Tests for ResultStore.resolve_ref method."""

    def test_resolve_by_id(self, tmp_path) -> None:
        """Given a numeric string ref, then the run with that id is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        result = store.resolve_ref(str(run_id))

        # Then
        assert result["id"] == run_id
        store.close()

    def test_resolve_by_tag(self, tmp_path) -> None:
        """Given a non-numeric tag, then the latest run with that tag is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        store.create_run("my-tag")
        id2 = store.create_run("my-tag")

        # When
        result = store.resolve_ref("my-tag")

        # Then
        assert result["id"] == id2  # latest
        store.close()

    def test_resolve_by_int_id(self, tmp_path) -> None:
        """Given an integer ref, then the run with that id is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        result = store.resolve_ref(run_id)

        # Then
        assert result["id"] == run_id
        store.close()

    def test_resolve_missing_raises(self, tmp_path) -> None:
        """Given a non-existent ref, then LookupError is raised."""
        # Given
        store = ResultStore(tmp_path / "test.db")

        # When / Then
        with pytest.raises(LookupError, match="No run found"):
            store.resolve_ref("nonexistent")
        store.close()

    def test_resolve_numeric_id_not_found_falls_through_to_tag(self, tmp_path) -> None:
        """Given a numeric id that doesn't exist, then tag lookup is attempted."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        store.create_run("999")  # tag happens to be numeric

        # When - "999" is digit, but no run with id=999; falls to tag lookup
        result = store.resolve_ref("999")

        # Then
        assert result["tag"] == "999"
        store.close()


class TestResultStoreGetResults:
    """Tests for ResultStore.get_results method."""

    def test_returns_empty_for_no_results(self, tmp_path) -> None:
        """Given a run with no results, then empty list is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")

        # When
        results = store.get_results(run_id)

        # Then
        assert results == []
        store.close()

    def test_returns_results_ordered_by_query_id(self, tmp_path) -> None:
        """Given multiple results, then they are ordered by query_id."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")
        for qid in ["q3", "q1", "q2"]:
            store.save_result(
                run_id=run_id,
                query_id=qid,
                category="A",
                k_value=10,
                retrieved_docs=[],
                ground_truth=[],
                recall_at_k=0.0,
                precision_at_k=0.0,
                latency_ms=1.0,
            )

        # When
        results = store.get_results(run_id)

        # Then
        query_ids = [r["query_id"] for r in results]
        assert query_ids == ["q1", "q2", "q3"]
        store.close()


class TestResultStoreListRuns:
    """Tests for ResultStore.list_runs method."""

    def test_returns_empty_when_no_runs(self, tmp_path) -> None:
        """Given no runs, then empty list is returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")

        # When
        runs = store.list_runs()

        # Then
        assert runs == []
        store.close()

    def test_returns_runs_newest_first(self, tmp_path) -> None:
        """Given multiple runs, then they are ordered newest first."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        id1 = store.create_run("first")
        id2 = store.create_run("second")

        # When
        runs = store.list_runs()

        # Then
        assert len(runs) == 2
        assert runs[0]["id"] == id2
        assert runs[1]["id"] == id1
        store.close()

    def test_includes_result_counts(self, tmp_path) -> None:
        """Given runs with results, then total and succeeded counts are included."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        run_id = store.create_run("test-run")
        store.save_result(
            run_id=run_id,
            query_id="q1",
            category="A",
            k_value=10,
            retrieved_docs=[],
            ground_truth=[],
            recall_at_k=1.0,
            precision_at_k=1.0,
            latency_ms=1.0,
            status="success",
        )
        store.save_result(
            run_id=run_id,
            query_id="q2",
            category="A",
            k_value=10,
            retrieved_docs=[],
            ground_truth=[],
            recall_at_k=None,
            precision_at_k=None,
            latency_ms=1.0,
            status="fatal",
        )

        # When
        runs = store.list_runs()

        # Then
        assert runs[0]["total"] == 2
        assert runs[0]["succeeded"] == 1
        store.close()

    def test_respects_limit(self, tmp_path) -> None:
        """Given a limit, then only that many runs are returned."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        for i in range(5):
            store.create_run(f"run-{i}")

        # When
        runs = store.list_runs(limit=2)

        # Then
        assert len(runs) == 2
        store.close()
