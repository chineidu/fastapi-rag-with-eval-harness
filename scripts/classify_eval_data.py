"""Classify FastAPI eval records into the specified labels.

Loads records from ``data/eval_dataset.jsonl``, classifies each via LLM
with few-shot prompting, and writes the labeled result to
``data/eval_dataset_labeled.jsonl``.

Usage:
    uv run python -m scripts.classify_eval_data
    uv run python -m scripts.classify_eval_data --help
"""

import asyncio
from pathlib import Path

import instructor
import openai
import typer
from pydantic import BaseModel, Field

from src import create_logger
from src.config import app_config, app_settings
from src.schemas.output import UnifiedEvalRecordSchema
from src.schemas.types import ClassificationLabel
from src.utils import read_jsonl, write_jsonl

logger = create_logger(name=__name__)

app = typer.Typer(
    help="Classify eval records into DIRECT_LOOKUP, MULTI_HOP, or CONCEPTUAL.",
    add_completion=False,
)


# 1. Create an OpenAI client targeting OpenRouter
openai_aclient = openai.AsyncOpenAI(
    base_url=app_settings.OPENROUTER_BASE_URL,
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
    timeout=app_config.eval_pipeline_config.classifier.timeout_seconds,
    max_retries=app_config.eval_pipeline_config.classifier.max_retries,
)
# 2. Patch it with Instructor
aclient = instructor.from_openai(openai_aclient)

# 3. Real few-shot examples sampled from data/eval_dataset.jsonl
_FEW_SHOT: list[dict[str, str | ClassificationLabel]] = [
    {
        "text": (
            "Title: How do I return an image in FastAPI?\n\n"
            "Body:\n"
            "Using the python module FastAPI, I can't figure out how to return an image. "
            "In flask I would do something like this:\n"
            '@app.route("/vector_image", methods=["POST"])\n'
            "def image_endpoint():\n"
            '    return Response(img, mimetype="image/png")\n\n'
            "what's the corresponding call in this module?"
        ),
        "label": ClassificationLabel.DIRECT_LOOKUP,
    },
    {
        "text": (
            "Title: How to add both file and JSON body in a FastAPI POST request?\n\n"
            "Body:\n"
            "I am trying to upload both a file and JSON data, as shown in the example "
            "below, but it is not working. If this is not the proper way for a POST "
            "request, please let me know how to select the required columns from an "
            "uploaded CSV file in FastAPI."
        ),
        "label": ClassificationLabel.MULTI_HOP,
    },
    {
        "text": (
            "Title: What are the best practices for structuring a FastAPI project?\n\n"
            "Body:\n"
            "The problem that I want to solve related the project setup:\n\n"
            "Good names of directories so that their purpose is clear.\n"
            "Keeping all project files (including virtualenv) in one place, so I\n"
            "can easily copy, move, archive, remove the whole project, or estimate "
            "disk space usage.\n"
            "Creating multiple copies of some selected file sets such as entire\n"
            "application, repository, or virtualenv, while keeping a single copy of\n"
            "other files that I don't want to clone.\n"
            "Deploying the right set of files to the server simply by resyncing\n"
            "selected one dir.\n"
            "handling both frontend and backend nicely."
        ),
        "label": ClassificationLabel.CONCEPTUAL,
    },
]

_SYSTEM_PROMPT = f"""\
You are a world-class text classification engine. Classify FastAPI support \
questions into one of three categories:

- DIRECT_LOOKUP: Answerable from a single concrete fact in the docs. The user \
asks "how do I do X" where X is a specific, well-documented feature.
- MULTI_HOP: Requires combining multiple features or concepts. The user needs \
to connect several pieces of information to solve their problem.
- CONCEPTUAL: About design, architecture, trade-offs, best practices, or \
opinions. Not tied to a single doc page.

Examples:

{"\n".join(f"Input:\n{ex['text']}\nLabel: {ex['label']}" for ex in _FEW_SHOT)}"""


class ClassificationResponse(BaseModel):
    """Predicted label for a single eval record, with chain-of-thought."""

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction. Max 2 sentences.",
    )
    label: ClassificationLabel = Field(
        ...,
        description="The predicted class label.",
    )


def _build_text(record: UnifiedEvalRecordSchema) -> str:
    """Build the classifier input from a record, truncating to MAX_INPUT_LENGTH."""
    text = f"Title: {record.title}\n\nBody:\n{record.body}"
    max_len = app_config.eval_pipeline_config.classifier.max_input_length
    if len(text) > max_len:
        text = text[:max_len]
    return text


async def _aclassify_single(record: UnifiedEvalRecordSchema) -> ClassificationResponse:
    """Perform single-label classification on a unified eval record."""
    text = _build_text(record)
    return await aclient.completions.create(
        model=app_config.eval_pipeline_config.classifier.model_id,
        response_model=ClassificationResponse,
        temperature=app_config.eval_pipeline_config.classifier.temperature,
        seed=app_config.eval_pipeline_config.classifier.seed,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Input:\n{text}\n\nLabel:",
            },
        ],
    )


async def aclassify_eval_data(
    records: list[UnifiedEvalRecordSchema],
) -> list[ClassificationResponse]:
    """Classify all records concurrently.

    Parameters
    ----------
    records : list[UnifiedEvalRecordSchema]
        Eval records to classify.

    Returns
    -------
    list[ClassificationResponse]
        One prediction per input record, in the same order.
    """
    tasks = [_aclassify_single(r) for r in records]
    return await asyncio.gather(*tasks)


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Log the invoked command before running it."""
    logger.info("running classify_eval_data command: %s", ctx.invoked_subcommand)


@app.command()
def classify(
    input_path: str = app_config.eval_pipeline_config.defaults.eval_dataset_path,
    output_path: str = app_config.eval_pipeline_config.defaults.eval_dataset_labeled_path,
) -> None:
    """Classify eval records and write the labeled result."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    records: list[UnifiedEvalRecordSchema] = read_jsonl(
        in_path, UnifiedEvalRecordSchema
    )
    logger.info("Loaded %d records from %s", len(records), in_path)

    preds = asyncio.run(aclassify_eval_data(records))

    # Update the label of each record with the predicted label
    labeled_records = [
        record.model_copy(update={"label": pred.label})
        for record, pred in zip(records, preds, strict=True)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, labeled_records)
    logger.info("Wrote %d labeled records to %s", len(labeled_records), out_path)


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
