"""Prompts for the ground-truth labeling pipeline."""

from src.schemas.types import ClassificationLabel

_CLASSIFY_FEW_SHOT: list[dict[str, str | ClassificationLabel]] = [
    {
        "text": """\
Title: How do I return an image in FastAPI?

Body:
Using the python module FastAPI, I can't figure out how to return an image.
In flask I would do something like this:
@app.route("/vector_image", methods=["POST"])
def image_endpoint():
    return Response(img, mimetype="image/png")

what's the corresponding call in this module?""",
        "label": ClassificationLabel.DIRECT_LOOKUP,
    },
    {
        "text": """\
Title: How to add both file and JSON body in a FastAPI POST request?

Body:
I am trying to upload both a file and JSON data, as shown in the example
below, but it is not working. If this is not the proper way for a POST
request, please let me know how to select the required columns from an
uploaded CSV file in FastAPI.""",
        "label": ClassificationLabel.MULTI_HOP,
    },
    {
        "text": """\
Title: What are the best practices for structuring a FastAPI project?

Body:
The problem that I want to solve related the project setup:

Good names of directories so that their purpose is clear.
Keeping all project files (including virtualenv) in one place, so I
can easily copy, move, archive, remove the whole project, or estimate
disk space usage.
Creating multiple copies of some selected file sets such as entire
application, repository, or virtualenv, while keeping a single copy of
other files that I don't want to clone.
Deploying the right set of files to the server simply by resyncing
selected one dir.
handling both frontend and backend nicely.""",
        "label": ClassificationLabel.CONCEPTUAL,
    },
]

CLASSIFY_SYSTEM_PROMPT = f"""\
You are a world-class text classification engine. Classify FastAPI support \
questions into one of three categories:

- DIRECT_LOOKUP: Answerable from a single concrete fact in the docs. The user \
asks "how do I do X" where X is a specific, well-documented feature.
- MULTI_HOP: Requires combining multiple features or concepts. The user needs \
to connect several pieces of information to solve their problem.
- CONCEPTUAL: About design, architecture, trade-offs, best practices, or \
opinions. Not tied to a single doc page.

Examples:

{"\n".join(f"Input:\n{ex['text']}\nLabel: {ex['label']}" for ex in _CLASSIFY_FEW_SHOT)}"""

JUDGE_SYSTEM_PROMPT = """\
You are a relevance judge for a RAG evaluation pipeline.

Given a user question, a reference answer, and a candidate document, determine
whether the document is relevant to answering the question.

A document is "relevant" if it contains information that would help answer the
question, even partially. A document is "irrelevant" if it does not contain
useful information for the question.

Respond with:
- verdict: "relevant" or "irrelevant"
- rationale: 1-2 sentences explaining your judgment
- confidence: a number between 0.0 and 1.0 indicating your confidence
"""
