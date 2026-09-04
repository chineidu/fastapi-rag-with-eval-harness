"""Tests for the eval_harness.loader module."""

import json

import pytest

from src.eval_harness.loader import GroundTruthLoader
from src.schemas.models import GroundTruthRecord

RECORD = {
    "query_id": "q1",
    "label": "DIRECT_LOOKUP",
    "query_text": "How to use FastAPI?",
    "relevant_docs": ["docs/quickstart.md"],
}


class TestGroundTruthLoaderLoad:
    """Tests for GroundTruthLoader.load static method."""

    def test_loads_json_array(self, tmp_path) -> None:
        """Given a JSON array file, then records are loaded in order."""
        # Given
        data_file = tmp_path / "ground_truth.json"
        records = [RECORD, {**RECORD, "query_id": "q2"}]
        data_file.write_text(json.dumps(records))

        # When
        result = GroundTruthLoader.load(data_file)

        # Then
        assert len(result) == 2
        assert result[0].query_id == "q1"
        assert result[1].query_id == "q2"

    def test_loads_jsonl(self, tmp_path) -> None:
        """Given a JSONL file, then records are loaded line by line."""
        # Given
        data_file = tmp_path / "ground_truth.jsonl"
        data_file.write_text(
            json.dumps(RECORD) + "\n" + json.dumps({**RECORD, "query_id": "q2"}) + "\n"
        )

        # When
        result = GroundTruthLoader.load(data_file)

        # Then
        assert len(result) == 2
        assert result[0].query_id == "q1"
        assert result[1].query_id == "q2"

    def test_skips_blank_lines_in_jsonl(self, tmp_path) -> None:
        """Given blank lines in JSONL, then they are skipped."""
        # Given
        data_file = tmp_path / "ground_truth.jsonl"
        data_file.write_text(
            json.dumps(RECORD)
            + "\n\n"
            + json.dumps({**RECORD, "query_id": "q2"})
            + "\n"
        )

        # When
        result = GroundTruthLoader.load(data_file)

        # Then
        assert len(result) == 2

    def test_empty_file_returns_empty_list(self, tmp_path) -> None:
        """Given an empty file, then an empty list is returned."""
        # Given
        data_file = tmp_path / "empty.json"
        data_file.write_text("")

        # When
        result = GroundTruthLoader.load(data_file)

        # Then
        assert result == []

    def test_whitespace_only_file_returns_empty_list(self, tmp_path) -> None:
        """Given a whitespace-only file, then an empty list is returned."""
        # Given
        data_file = tmp_path / "whitespace.json"
        data_file.write_text("   \n  \n  ")

        # When
        result = GroundTruthLoader.load(data_file)

        # Then
        assert result == []

    def test_missing_file_raises(self) -> None:
        """Given a non-existent path, then FileNotFoundError is raised."""
        # Given / When / Then
        with pytest.raises(FileNotFoundError):
            GroundTruthLoader.load("/nonexistent/path.json")

    def test_duplicate_query_id_raises(self, tmp_path) -> None:
        """Given duplicate query_ids, then ValueError is raised."""
        # Given
        data_file = tmp_path / "dups.json"
        records = [RECORD, RECORD]
        data_file.write_text(json.dumps(records))

        # When / Then
        with pytest.raises(ValueError, match="Duplicate query_id"):
            GroundTruthLoader.load(data_file)

    def test_invalid_record_raises(self, tmp_path) -> None:
        """Given a record failing validation, then ValueError is raised."""
        # Given
        data_file = tmp_path / "invalid.json"
        bad_record = {
            "query_id": "",
            "label": "",
            "query_text": "",
            "relevant_docs": [],
        }
        data_file.write_text(json.dumps([bad_record]))

        # When / Then
        with pytest.raises(ValueError, match="Invalid ground truth record"):
            GroundTruthLoader.load(data_file)

    def test_invalid_json_raises(self, tmp_path) -> None:
        """Given invalid JSON content, then ValueError is raised."""
        # Given
        data_file = tmp_path / "bad.json"
        data_file.write_text("{invalid json")

        # When / Then
        with pytest.raises(ValueError, match="Invalid JSON"):
            GroundTruthLoader.load(data_file)

    def test_invalid_jsonl_line_raises(self, tmp_path) -> None:
        """Given an invalid JSON line in JSONL, then ValueError is raised."""
        # Given
        data_file = tmp_path / "bad.jsonl"
        data_file.write_text(json.dumps(RECORD) + "\n{bad json\n")

        # When / Then
        with pytest.raises(ValueError, match="Invalid JSON on line"):
            GroundTruthLoader.load(data_file)

    def test_loads_as_ground_truth_record_instances(self, tmp_path) -> None:
        """Given valid records, then they are GroundTruthRecord instances."""
        # Given
        data_file = tmp_path / "gt.json"
        data_file.write_text(json.dumps([RECORD]))

        # When
        result = GroundTruthLoader.load(data_file)

        # Then
        assert isinstance(result[0], GroundTruthRecord)

    def test_string_path_converted(self, tmp_path) -> None:
        """Given a string path, then it is accepted and loaded."""
        # Given
        data_file = tmp_path / "gt.json"
        data_file.write_text(json.dumps([RECORD]))

        # When
        result = GroundTruthLoader.load(str(data_file))

        # Then
        assert len(result) == 1
