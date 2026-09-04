"""SQLite persistence for eval runs and per-query results."""

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from src import create_logger
from src.schemas.types import QueryResultStatus, RunStatus

logger = create_logger(name=__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tag         TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    status      TEXT NOT NULL DEFAULT 'running',
    config      TEXT,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS query_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES runs(id),
    query_id           TEXT NOT NULL,
    category           TEXT NOT NULL,
    k_value            INTEGER NOT NULL,
    retrieved_docs     TEXT NOT NULL,
    ground_truth       TEXT NOT NULL,
    recall_at_k        REAL,
    precision_at_k     REAL,
    latency_ms         REAL,
    status             TEXT NOT NULL DEFAULT 'success',
    error              TEXT,
    metrics_extra      TEXT
);

CREATE INDEX IF NOT EXISTS idx_query_results_run_query
    ON query_results(run_id, query_id);
"""


class ResultStore:
    """Thread-safe SQLite store for runs and per-query results.

    A single connection is shared across the runner's worker threads and
    guarded by a lock, since ``ThreadPoolExecutor`` workers write
    concurrently.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Open (creating parents) and initialize the SQLite database.

        Parameters
        ----------
        db_path : Path | str
            Path to the SQLite file.

        """
        self.db_path = Path(db_path)
        if self.db_path.parent != Path():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        """Return this store for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the connection when leaving a ``with`` block."""
        self.close()

    def create_run(
        self,
        tag: str,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert a new run row with status ``running`` and return its id.

        Parameters
        ----------
        tag : str
            Non-unique label describing this run.
        config : Mapping[str, Any] | None
            JSON-serializable run config (k, concurrency, adapter, ...).
        metadata : Mapping[str, Any] | None
            Freeform JSON-serializable notes.

        Returns
        -------
        int
            The new run id.

        """
        config_json = json.dumps(dict(config)) if config is not None else None
        metadata_json = json.dumps(dict(metadata)) if metadata is not None else None
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO runs (tag, status, config, metadata) VALUES (?, ?, ?, ?);",
                (tag, RunStatus.RUNNING.value, config_json, metadata_json),
            )
            self._conn.commit()
            last_id = cursor.lastrowid
        if last_id is None:
            raise RuntimeError("Failed to create run: no row id returned")
        logger.info("Created run %s with tag %r", last_id, tag)
        return last_id

    def save_result(
        self,
        run_id: int,
        query_id: str,
        category: str,
        k_value: int,
        retrieved_docs: Sequence[Sequence[object]],
        ground_truth: Sequence[str],
        recall_at_k: float | None,
        precision_at_k: float | None,
        latency_ms: float,
        status: str = QueryResultStatus.SUCCESS.value,
        error: str | None = None,
        metrics_extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist one per-query result row.

        Parameters
        ----------
        run_id : int
            Owning run id.
        query_id : str
            Ground truth query id.
        category : str
            Difficulty category (drives per-category breakdown).
        k_value : int
            Retrieval depth used for the metrics.
        retrieved_docs : Sequence[Sequence[object]]
            Full ranked ``(doc_path, score)`` list, stored as JSON.
        ground_truth : Sequence[str]
            Relevant doc paths, stored as JSON.
        recall_at_k : float | None
            Recall value, or ``None`` when the query failed.
        precision_at_k : float | None
            Precision value, or ``None`` when the query failed.
        latency_ms : float
            Adapter call latency in milliseconds.
        status : str
            One of ``success``, ``recoverable``, ``fatal``.
        error : str | None
            Error message when the query failed.
        metrics_extra : Mapping[str, Any] | None
            Catch-all JSON for experimental metrics.

        """
        retrieved_json = json.dumps([[doc, score] for doc, score in retrieved_docs])
        ground_truth_json = json.dumps(list(ground_truth))
        extra_json = (
            json.dumps(dict(metrics_extra)) if metrics_extra is not None else None
        )
        with self._lock:
            self._conn.execute(
                """INSERT INTO query_results
                   (run_id, query_id, category, k_value, retrieved_docs, ground_truth,
                    recall_at_k, precision_at_k, latency_ms, status, error, metrics_extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (
                    run_id,
                    query_id,
                    category,
                    k_value,
                    retrieved_json,
                    ground_truth_json,
                    recall_at_k,
                    precision_at_k,
                    latency_ms,
                    status,
                    error,
                    extra_json,
                ),
            )
            self._conn.commit()

    def finalize_run(self, run_id: int, status: str) -> None:
        """Mark a run ``complete`` or ``partial``.

        Parameters
        ----------
        run_id : int
            Run to finalize.
        status : str
            One of ``complete`` or ``partial``.

        """
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status = ? WHERE id = ?;", (status, run_id)
            )
            self._conn.commit()
        logger.info("Finalized run %d as %s", run_id, status)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Return a run row as a dict, or ``None`` when missing.

        Parameters
        ----------
        run_id : int
            Run id to fetch.

        Returns
        -------
        dict[str, Any] | None
            Row mapping, or ``None``.

        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?;", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def resolve_ref(self, ref: str | int) -> dict[str, Any]:
        """Resolve a ``TAG|ID`` reference to a run row.

        Numeric references match a run id; anything else resolves to the
        latest run with that tag. Use a numeric id for precise comparison
        when tags are shared.

        Parameters
        ----------
        ref : str | int
            Run id or tag.

        Returns
        -------
        dict[str, Any]
            The resolved run row.

        Raises
        ------
        LookupError
            If no run matches ``ref``.

        """
        text = str(ref)
        with self._lock:
            if text.isdigit():
                # Use the id
                row = self._conn.execute(
                    "SELECT * FROM runs WHERE id = ?", (int(text),)
                ).fetchone()
                if row is not None:
                    return dict(row)

            # Use the tag
            row = self._conn.execute(
                "SELECT * FROM runs WHERE tag = ? ORDER BY id DESC LIMIT 1;", (text,)
            ).fetchone()
        if row is None:
            raise LookupError(f"No run found for {ref!r}")
        return dict(row)

    def get_results(self, run_id: int) -> list[dict[str, Any]]:
        """Return all per-query result rows for a run, ordered by query id.

        Parameters
        ----------
        run_id : int
            Run id to fetch results for.

        Returns
        -------
        list[dict[str, Any]]
            Row mappings in ``query_id`` order.

        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM query_results WHERE run_id = ? ORDER BY query_id;",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent runs with success/total query counts.

        Parameters
        ----------
        limit : int
            Maximum rows to return.

        Returns
        -------
        list[dict[str, Any]]
            Newest-first rows with ``id``, ``tag``, ``created_at``,
            ``status``, ``total``, and ``succeeded`` keys.

        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.id, r.tag, r.created_at, r.status,
                          COUNT(q.id) AS total,
                          COALESCE(SUM(CASE WHEN q.status = 'success' THEN 1 ELSE 0 END), 0)
                            AS succeeded
                   FROM runs r LEFT JOIN query_results q ON q.run_id = r.id
                   GROUP BY r.id ORDER BY r.id DESC LIMIT ?;""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
