import logging
from pathlib import Path
from typing import Any

import pytest

from src.schemas.output import DiscussionNodeSchema
from src.utils.io import read_jsonl, write_jsonl


@pytest.fixture
def valid_record() -> dict[str, Any]:
    return {
        "id": "D_abc",
        "number": 1,
        "title": "Test",
        "url": "https://example.com/1",
        "body": "<p>body</p>",
        "bodyText": "body",
        "createdAt": "2024-01-01T00:00:00Z",
        "closedAt": None,
        "answerChosenAt": None,
        "upvoteCount": 5,
        "isAnswered": True,
        "stateReason": "RESOLVED",
        "category": {"name": "Q&A", "slug": "questions"},
        "labels": {"nodes": []},
        "comments": {"totalCount": 0},
        "answer": None,
    }


def _write_lines(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestReadJsonl:
    def test_reads_valid_records(
        self, tmp_path: Path, valid_record: dict[str, Any]
    ) -> None:
        # Given
        import json

        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
        # When
        records = read_jsonl(path, DiscussionNodeSchema)
        # Then
        assert len(records) == 1
        assert records[0].id == "D_abc"
        assert records[0].number == 1
        assert records[0].title == "Test"

    def test_skips_empty_lines(
        self, tmp_path: Path, valid_record: dict[str, Any]
    ) -> None:
        # Given
        import json

        path = tmp_path / "test.jsonl"
        path.write_text(
            "\n" + json.dumps(valid_record) + "\n\n" + json.dumps(valid_record) + "\n",
            encoding="utf-8",
        )
        # When
        records = read_jsonl(path, DiscussionNodeSchema)
        # Then
        assert len(records) == 2

    def test_skips_malformed_lines(
        self,
        tmp_path: Path,
        valid_record: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Given
        import json

        path = tmp_path / "test.jsonl"
        path.write_text(
            json.dumps(valid_record)
            + "\n"
            + "not valid json\n"
            + json.dumps(valid_record)
            + "\n",
            encoding="utf-8",
        )
        caplog.set_level(logging.WARNING)
        # When
        records = read_jsonl(path, DiscussionNodeSchema)
        # Then
        assert len(records) == 2
        assert any("Skipping invalid record" in r.message for r in caplog.records)

    def test_skips_line_with_msgspec_error_when_field_missing(
        self, tmp_path: Path, valid_record: dict[str, Any]
    ) -> None:
        # Given
        import json

        path = tmp_path / "test.jsonl"
        incomplete = valid_record.pop("id")
        _ = incomplete  # ensure pop happened
        path.write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
        # When
        records = read_jsonl(path, DiscussionNodeSchema)
        # Then
        assert len(records) == 0

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        # Given
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        # When
        records = read_jsonl(path, DiscussionNodeSchema)
        # Then
        assert records == []


class TestWriteJsonl:
    def test_roundtrip(self, tmp_path: Path, valid_record: dict[str, Any]) -> None:
        # Given
        records = [DiscussionNodeSchema.model_validate(valid_record)]
        path = tmp_path / "roundtrip.jsonl"
        # When
        write_jsonl(path, records)
        loaded = read_jsonl(path, DiscussionNodeSchema)
        # Then
        assert len(loaded) == 1
        assert loaded[0].id == records[0].id

    def test_output_uses_camel_case_keys(
        self, tmp_path: Path, valid_record: dict[str, Any]
    ) -> None:
        # Given
        records = [DiscussionNodeSchema.model_validate(valid_record)]
        path = tmp_path / "camel.jsonl"
        # When
        write_jsonl(path, records)
        content = path.read_text(encoding="utf-8")
        # Then
        assert '"id"' in content
        assert '"createdAt"' in content
        assert '"bodyText"' in content

    def test_writes_multiple_records(
        self, tmp_path: Path, valid_record: dict[str, Any]
    ) -> None:
        # Given
        records = [DiscussionNodeSchema.model_validate(valid_record) for _ in range(3)]
        path = tmp_path / "multi.jsonl"
        # When
        write_jsonl(path, records)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        # Then
        assert len(lines) == 3
