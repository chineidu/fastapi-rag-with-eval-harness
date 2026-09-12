"""Ground truth loading and validation for the eval harness."""

import json
from pathlib import Path

from src import create_logger
from src.schemas.models import GroundTruthRecord

logger = create_logger(name=__name__)


class GroundTruthLoader:
    """Load and validate a ground truth file.

    Accepts either a JSON array (legacy, one object per query) or JSONL
    (one object per line, the format in ADR-0008).
    """

    @staticmethod
    def load(path: Path | str) -> list[GroundTruthRecord]:
        """Load ground truth records from a JSON or JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to the ground truth file.

        Returns
        -------
        list[GroundTruthRecord]
            Validated records, in file order.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file is not valid JSON/JSONL, a record fails validation,
            or two records share the same ``query_id``.

        """
        ground_truth_path = Path(path)
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

        text = ground_truth_path.read_text(encoding="utf-8")
        raw_records = GroundTruthLoader._parse_text(text, ground_truth_path)

        records: list[GroundTruthRecord] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_records):
            try:
                record = GroundTruthRecord.model_validate(raw)
            except ValueError as e:
                raise ValueError(
                    f"Invalid ground truth record at index {index} "
                    f"in {ground_truth_path}: {e}"
                ) from e
            if record.query_id in seen:
                raise ValueError(
                    f"Duplicate query_id {record.query_id!r} in {ground_truth_path}"
                )
            seen.add(record.query_id)
            records.append(record)

        logger.info(
            "Loaded %d ground truth records from %s", len(records), ground_truth_path
        )
        return records

    @staticmethod
    def _parse_text(text: str, path: Path) -> list[object]:
        """Parse file text as a JSON array or as JSONL.

        Parameters
        ----------
        text : str
            Raw file content.
        path : Path
            Source path, used in error messages.

        Returns
        -------
        list[object]
            Raw record dicts.

        Raises
        ------
        ValueError
            If the content is neither a JSON array nor valid JSONL.

        """
        stripped = text.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON array in {path}: {e}") from e
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {path}")
            return data
        records: list[object] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {path}: {e}"
                ) from e
        return records
