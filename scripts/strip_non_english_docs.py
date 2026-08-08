"""Remove all non-English language directories from the FastAPI docs corpus."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs/fastapi/docs")
KEEP = {"en"}


def main() -> None:
    """Remove all non-English language directories from the FastAPI docs corpus."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not DOCS_DIR.is_dir():
        logger.error("Directory not found: %s", DOCS_DIR)
        raise SystemExit(1)

    removed = 0
    for entry in sorted(DOCS_DIR.iterdir()):
        if entry.is_dir() and entry.name not in KEEP:
            shutil.rmtree(entry)
            logger.info("Removed %s", entry)
            removed += 1

    logger.info("Done. Removed %d directories.", removed)


if __name__ == "__main__":
    main()
# To use, run: uv run python scripts/strip_non_english_docs.py
