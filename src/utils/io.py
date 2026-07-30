import json
from collections.abc import Sequence
from pathlib import Path

import msgspec
from pydantic import BaseModel

from src import create_logger

logger = create_logger(name=__name__)


def read_jsonl[M: BaseModel](path: Path, schema: type[M]) -> list[M]:
    """Load and validate records from a JSONL file."""
    records: list[M] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = msgspec.json.decode(line)
                records.append(schema.model_validate(data))
            except (msgspec.DecodeError, ValueError) as e:
                logger.warning(
                    "Skipping invalid record at %s:%d — %s", path, line_no, e
                )
    return records


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    """Write Pydantic records to a JSONL file, one per line."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                f"{json.dumps(record.model_dump(by_alias=True), ensure_ascii=False, default=str)}\n"
            )
