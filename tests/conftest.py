import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _import_script(script_name: str) -> Any:
    """Import a scripts/*.py module by file path without touching scripts/__init__.py."""
    root = Path(__file__).parent.parent
    path = root / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fetch_eval_data():
    """Import scripts/fetch_eval_data.py as a module."""
    return _import_script("fetch_eval_data")


@pytest.fixture(scope="session")
def normalize_eval_data():
    """Import scripts/normalize_eval_data.py as a module."""
    return _import_script("normalize_eval_data")


@pytest.fixture
def github_node_payload() -> dict[str, Any]:
    return {
        "id": "D_kwDOAFa5CM4A2xYQ",
        "number": 42,
        "title": "How to use FastAPI dependencies?",
        "url": "https://github.com/fastapi/fastapi/discussions/42",
        "body": "<p>Question body with <code>code</code></p>",
        "bodyText": "Question body with code",
        "createdAt": "2024-01-15T10:30:00Z",
        "closedAt": "2024-01-20T15:00:00Z",
        "answerChosenAt": "2024-01-18T12:00:00Z",
        "upvoteCount": 15,
        "isAnswered": True,
        "stateReason": "RESOLVED",
        "category": {"name": "Questions", "slug": "questions"},
        "labels": {"nodes": [{"name": "question"}, {"name": "dependencies"}]},
        "comments": {"totalCount": 3},
        "answer": {
            "id": "A_kw002",
            "url": "https://github.com/fastapi/fastapi/discussions/42#answer-1",
            "body": "<p>Use <code>Depends()</code></p>",
            "bodyText": "Use Depends()",
            "createdAt": "2024-01-16T08:00:00Z",
            "upvoteCount": 12,
            "isAnswer": True,
            "author": {"login": "tiangolo", "url": "https://github.com/tiangolo"},
            "reactionGroups": [
                {"content": "THUMBS_UP", "reactors": {"totalCount": 12}}
            ],
        },
    }


@pytest.fixture
def stackoverflow_question_payload() -> dict[str, Any]:
    return {
        "question_id": 12345,
        "title": "FastAPI middleware not working",
        "link": "https://stackoverflow.com/q/12345",
        "body": "<p>I have a custom middleware but it doesn't work.</p>",
        "body_markdown": "I have a custom middleware but it doesn't work.",
        "score": 25,
        "answer_count": 3,
        "view_count": 1500,
        "creation_date": 1705312800,
        "tags": ["fastapi", "middleware", "python"],
        "is_answered": True,
        "answer": {
            "answer_id": 67890,
            "question_id": 12345,
            "score": 30,
            "is_accepted": True,
            "body": "<p>Make sure to add middleware before routes.</p>",
            "body_markdown": "Make sure to add middleware before routes.",
            "creation_date": 1705399200,
            "link": "https://stackoverflow.com/a/67890",
        },
    }
