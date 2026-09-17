"""Bundled static chat page for the RAG service."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", include_in_schema=False)
async def ui_index() -> FileResponse:
    """Return the bundled chat page."""
    return FileResponse(_INDEX_HTML)
