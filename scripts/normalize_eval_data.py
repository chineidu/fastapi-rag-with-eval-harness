from datetime import UTC, datetime
from pathlib import Path

import typer

from src import create_logger
from src.config import app_config
from src.schemas.output import (
    DiscussionNodeSchema,
    StackOverflowQuestionSchema,
    UnifiedEvalRecordSchema,
)
from src.utils import read_jsonl, strip_html, write_jsonl

logger = create_logger(name=__name__)

app = typer.Typer(
    help="Normalize fetched data into a unified eval dataset", add_completion=False
)


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Log the invoked command before running it."""
    logger.info("running normalize_eval_data command: %s", ctx.invoked_subcommand)


def _github_to_unified(rec: DiscussionNodeSchema) -> UnifiedEvalRecordSchema:
    """Convert a GitHub discussion to a unified eval record."""
    answer = rec.answer or {}
    return UnifiedEvalRecordSchema(
        id=str(rec.id),
        source="github",
        title=rec.title,
        url=rec.url,
        body=rec.body,
        answer_text=answer.get("body", ""),
        created_at=rec.created_at,
        score=rec.upvote_count,
        answer_score=answer.get("upvoteCount", 0),
        tags=[n["name"] for n in rec.labels.get("nodes", [])],
    )


def _stackoverflow_to_unified(
    rec: StackOverflowQuestionSchema,
) -> UnifiedEvalRecordSchema:
    """Convert a Stack Overflow question to a unified eval record."""
    answer = rec.answer
    if answer is None:
        answer_text = ""
        answer_score = 0
    else:
        answer_text = answer.body_markdown or strip_html(answer.body)
        answer_score = answer.score
    return UnifiedEvalRecordSchema(
        id=str(rec.question_id),
        source="stackoverflow",
        title=rec.title,
        url=rec.link,
        body=rec.body_markdown or strip_html(rec.body),
        answer_text=answer_text,
        created_at=datetime.fromtimestamp(rec.creation_date, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        score=rec.score,
        answer_score=answer_score,
        tags=rec.tags,
    )


@app.command()
def normalize(
    github: str = app_config.eval_pipeline_config.defaults.github_discussions_path,
    stackoverflow: str = app_config.eval_pipeline_config.defaults.stackoverflow_questions_path,
    output: str = app_config.eval_pipeline_config.defaults.eval_dataset_path,
) -> None:
    """Combine GitHub and Stack Overflow datasets into a unified eval JSONL."""
    unified: list[UnifiedEvalRecordSchema] = []

    github_path = Path(github)
    if github_path.exists():
        gh = read_jsonl(github_path, DiscussionNodeSchema)
        unified.extend(_github_to_unified(r) for r in gh)
        logger.info("Converted %d GitHub discussions", len(gh))
    else:
        logger.warning("GitHub file not found: %s", github)

    so_path = Path(stackoverflow)
    if so_path.exists():
        so = read_jsonl(so_path, StackOverflowQuestionSchema)
        unified.extend(_stackoverflow_to_unified(r) for r in so)
        logger.info("Converted %d Stack Overflow questions", len(so))
    else:
        logger.warning("Stack Overflow file not found: %s", stackoverflow)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, unified)
    logger.info("Wrote %d unified records to %s", len(unified), output)


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
