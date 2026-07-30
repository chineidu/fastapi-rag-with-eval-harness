import shutil
from pathlib import Path

DOCS_DIR = Path("docs/fastapi/docs")
KEEP = {"en"}


def main() -> None:
    """Remove all non-English language directories from the FastAPI docs corpus."""
    if not DOCS_DIR.is_dir():
        print(f"Directory not found: {DOCS_DIR}")
        raise SystemExit(1)

    removed = 0
    for entry in sorted(DOCS_DIR.iterdir()):
        if entry.is_dir() and entry.name not in KEEP:
            shutil.rmtree(entry)
            print(f"Removed {entry}")
            removed += 1

    print(f"Done. Removed {removed} directories.")


if __name__ == "__main__":
    main()
# To use, run: uv run python scripts/strip_non_english_docs.py
