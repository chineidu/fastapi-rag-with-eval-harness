"""Prompts for Slice 2 grounded generation (ADR-0027)."""

from src.schemas.retrieval import SearchHit

ABSTENTION_ANSWER = "I don't know based on the provided docs."

GENERATE_SYSTEM_PROMPT = f"""\
<role>
    You are a FastAPI documentation assistant.
    Answer questions using only the retrieved documentation provided.
</role>

<instructions>
    Follow these steps in order:
    1. Read the <question>, then read each <document> in <documents>.
    2. Ground every factual claim in the document contents.
    Your response will be checked against the cited sources.
    3. Cite the source path in square brackets after each claim, e.g.:
    [docs/fastapi/docs/en/docs/tutorial/body.md]
    Only cite paths that appear in <source> tags.
    4. If the documents do not contain the answer, reply with exactly:
    {ABSTENTION_ANSWER}
</instructions>

<examples>
    <example>
        <question>How do I return an image in FastAPI?</question>
        <answer>Use a Response with media_type set to image/png
        [docs/fastapi/docs/en/docs/advanced/custom-response.md].</answer>
    </example>
    <example>
        <question>How do I deploy FastAPI to the moon?</question>
        <answer>{ABSTENTION_ANSWER}</answer>
    </example>
</examples>"""


def format_generation_context(contexts: list[SearchHit]) -> str:
    """Format ranked chunk hits as a tagged documents block.

    Parameters
    ----------
    contexts : list[SearchHit]
        Ranked chunk hits in relevance order. Passed through without
        truncation in v1 (ADR-0027).

    Returns
    -------
    str
        A `<documents>` block with one `<document>` per chunk carrying
        `<source>` and `<document_content>` subtags. Empty input returns
        an empty `<documents>` block.

    """
    if not contexts:
        return "<documents>\n</documents>"
    blocks: list[str] = [
        f'<document index="{index}">\n<source>{hit.doc_path}</source>\n'
        f"<document_content>\n{hit.text}\n</document_content>\n</document>"
        for index, hit in enumerate(contexts, start=1)
    ]
    return "<documents>\n" + "\n".join(blocks) + "\n</documents>"
