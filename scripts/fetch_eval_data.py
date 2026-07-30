import asyncio
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import typer

from src import create_logger
from src.config import app_config, app_settings
from src.schemas.output import (
    DiscussionNodeSchema,
    StackOverflowAnswerSchema,
    StackOverflowQuestionSchema,
)
from src.schemas.types import RepoHandle
from src.utils import write_jsonl

logger = create_logger(name=__name__)

type OutputRecord = DiscussionNodeSchema | StackOverflowQuestionSchema


def _parse_repo_url(url: str) -> RepoHandle:
    """Extract owner and repo from a GitHub URL like https://github.com/owner/repo."""
    parsed = urllib.parse.urlparse(url)
    # Remove trailing slash and .git suffix
    path = parsed.path.rstrip("/").removesuffix(".git")
    # Remove leading slash
    path = path.lstrip("/")
    # Split path by slash
    parts = path.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Expected GitHub repo URL (e.g. https://github.com/owner/repo), got: {url}"
        )
    return RepoHandle(owner=parts[0], name=parts[1])


# GraphQL queries look like JSON but define *what* fields you want back.
# Variables (`$name: String!`) are passed separately in the POST body,
# like function arguments. `!` means the value is required.
#
# This query fetches up to 10 discussion categories for a repo (e.g.
# "Q&A", "Ideas"). Each category has an opaque `id` and a human-readable
# `slug`. The larger DISCUSSIONS_QUERY requires a category *ID* to
# paginate, but callers provide a *slug* — so we map slug → ID here.
CATEGORY_ID_QUERY: str = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussionCategories(first: 10) {
      nodes {
        id
        slug
      }
    }
  }
}
"""


DISCUSSIONS_QUERY: str = """
query($owner: String!, $name: String!, $cursor: String, $category_id: ID!, $first: Int!) {
  repository(owner: $owner, name: $name) {
    discussions(
      first: $first
      after: $cursor
      categoryId: $category_id
      orderBy: { field: UPDATED_AT, direction: DESC }
    ) {
      totalCount
      nodes {
        id
        number
        title
        url
        body
        bodyHTML
        bodyText
        createdAt
        updatedAt
        closedAt
        answerChosenAt
        upvoteCount
        isAnswered
        stateReason
        locked
        closed
        category {
          name
          slug
        }
        labels(first: 10) {
          nodes {
            name
          }
        }
        reactionGroups {
          content
          reactors {
            totalCount
          }
        }
        comments {
          totalCount
        }
        answer {
          id
          url
          body
          bodyHTML
          bodyText
          createdAt
          upvoteCount
          isAnswer
          author {
            login
            url
          }
          reactionGroups {
            content
            reactors {
              totalCount
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""


def _get_github_auth_headers() -> dict[str, str]:
    """Build auth headers for GitHub GraphQL API."""
    token = app_settings.GITHUB_READ_ACCESS.get_secret_value()
    if not token:
        raise RuntimeError("GITHUB_READ_ACCESS environment variable is not set")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def _run_query(
    client: httpx.AsyncClient,
    *,
    query: str,
    variables: dict[str, Any],
) -> dict[str, Any]:
    """Execute a GraphQL query against the GitHub API with retry on rate limit."""
    headers = _get_github_auth_headers()
    for attempt in range(app_config.eval_pipeline_config.github.max_retries):
        # Make a POST request to GitHub GraphQL API
        response = await client.post(
            app_config.eval_pipeline_config.github.graphql_url,
            headers=headers,
            json={"query": query, "variables": variables},
        )
        # Handle rate limiting by retrying with exponential backoff
        if response.status_code == 429:
            retry_after = int(response.headers.get("retry-after", 10))
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %ds",
                attempt + 1,
                app_config.eval_pipeline_config.github.max_retries,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            continue
        # If the response is not OK, raise an error
        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned HTTP {response.status_code}: {response.text}"
            )
        # Parse the response JSON
        data = response.json()
        # Check for GraphQL errors
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        return data
    # If max retries are exceeded, raise an error
    raise RuntimeError("Max retries exceeded for GitHub API rate limiting")


async def _resolve_category_id(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    slug: str,
) -> str:
    """Resolve discussion category ID from slug."""
    # Fetch the available discussion categories
    data: dict[str, Any] = await _run_query(
        client, query=CATEGORY_ID_QUERY, variables={"owner": owner, "name": repo}
    )
    # Extract the categories from the response
    categories: list[dict[str, Any]] = data["data"]["repository"][
        "discussionCategories"
    ]["nodes"]
    # Iterate over the categories to find the one with the matching slug
    for cat in categories:
        if cat["slug"] == slug:
            return cat["id"]
    # If the requested category slug is not found, inform the user
    available: list[str] = [c["slug"] for c in categories]
    raise RuntimeError(
        f"Category '{slug}' not found in {owner}/{repo}. Available: {available}"
    )


def _is_answered_and_resolved(node: DiscussionNodeSchema) -> bool:
    """Check if a discussion is answered and resolved."""
    return bool(node.is_answered) and node.state_reason == "RESOLVED"


async def afetch_data_from_github(
    url: str,
    num_issues: int,
    output_path: str,
    category_slug: str = "questions",
) -> None:
    """Fetch answered and resolved questions with metadata from GitHub using GraphQL API.

    Parses the repo URL to determine owner/repo, resolves the discussion category by slug,
    paginates through discussions filtered by ``isAnswered`` and ``stateReason: RESOLVED``,
    and writes the raw API response as JSONL to ``output_path``.
    """
    owner, repo = _parse_repo_url(url)
    logger.info(
        "Fetching %d answered discussions from %s/%s (category=%s)",
        num_issues,
        owner,
        repo,
        category_slug,
    )

    async with httpx.AsyncClient(timeout=30) as client:
        # Resolve the category ID for the given category slug
        category_id = await _resolve_category_id(client, owner, repo, category_slug)
        logger.info("Resolved category ID for '%s'", category_slug)

        matched: list[DiscussionNodeSchema] = []
        cursor: str | None = None

        # Start Fetching discussions using pagination: Fetch only the required number of discussions
        while len(matched) < num_issues:
            # Fetch discussions for the current category
            data: dict[str, Any] = await _run_query(
                client,
                query=DISCUSSIONS_QUERY,
                variables={
                    "owner": owner,
                    "name": repo,
                    "cursor": cursor,
                    "category_id": category_id,
                    "first": app_config.eval_pipeline_config.github.page_size,
                },
            )
            page = data["data"]["repository"]["discussions"]
            parsed: list[DiscussionNodeSchema] = [
                DiscussionNodeSchema.model_validate(n) for n in page["nodes"]
            ]
            # Select ONLY the nodes that meet the condition
            matched.extend(n for n in parsed if _is_answered_and_resolved(n))

            logger.info(
                "Collected %d/%d answered+resolved discussions",
                len(matched),
                num_issues,
            )

            page_info = page["pageInfo"]
            if not page_info["hasNextPage"]:
                logger.info(
                    "No more pages available (%d total discussions in category)",
                    page["totalCount"],
                )
                break

            cursor = page_info["endCursor"]
            await asyncio.sleep(app_config.eval_pipeline_config.github.retry_sleep_secs)

    # Ensure ONLY the required number of discussions are selected
    matched = matched[:num_issues]
    # Sort discussions by upvote_count
    matched.sort(key=lambda n: n.upvote_count, reverse=True)

    output = Path(output_path)
    # Create the output directory if it does not exist
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write the matched discussions to a JSONL file
    await asyncio.to_thread(write_jsonl, output, matched)
    logger.info("Wrote %d discussions to %s", len(matched), output)


def _parse_stack_exchange_url(url: str) -> str:
    """Extract site name from a Stack Exchange URL (e.g. ``stackoverflow.com`` → ``stackoverflow``)."""
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        # Return the stripped URL if hostname is not detected
        return url.rstrip("/")
    hostname = parsed.hostname.lower()
    return hostname.split(".")[0]


def _pick_best_answer(answers: list[dict[str, Any]]) -> StackOverflowAnswerSchema:
    """Return accepted answer, or highest-scored answer if none is accepted."""
    target = next((a for a in answers if a.get("is_accepted")), None) or max(
        answers, key=lambda a: a.get("score", 0)
    )
    if "link" not in target:
        target["link"] = f"https://stackoverflow.com/a/{target['answer_id']}"
    return StackOverflowAnswerSchema.model_validate(target)


async def _fetch_stack_exchange_page(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fetch a single page from the Stack Exchange API, respecting backoff."""
    response = await client.get(endpoint, params=params)

    # Handle HTTP errors
    if response.status_code != 200:
        raise RuntimeError(
            f"Stack Exchange API returned HTTP {response.status_code}: {response.text}"
        )

    # Handle Stack Exchange API errors
    data = response.json()
    if "error_id" in data:
        raise RuntimeError(
            f"Stack Exchange API error {data['error_id']}: {data.get('error_message', '')}"
        )

    # Respect rate limiting backoff
    backoff = data.get("backoff", 0)
    if backoff:
        await asyncio.sleep(backoff)

    return data


async def afetch_stack_exchange_data(
    url: str,
    num_issues: int,
    output_path: str,
    tag: str = "fastapi",
) -> None:
    """Fetch answered questions from Stack Exchange by tag and write to JSONL."""
    site = _parse_stack_exchange_url(url)
    logger.info(
        "Fetching %d answered questions from %s (tag=%s)",
        num_issues,
        site,
        tag,
    )

    api_key = app_settings.STACK_EXCHANGE_READ_ACCESS.get_secret_value()

    matched: list[OutputRecord] = []
    page = 1

    async with httpx.AsyncClient(timeout=30) as client:
        while len(matched) < num_issues:
            base_params: dict[str, Any] = {
                "site": site,
                "pagesize": app_config.eval_pipeline_config.stack_exchange.page_size,
                "sort": "votes",
                "order": "desc",
                "filter": "withbody",
            }
            if api_key:
                base_params["key"] = api_key

            questions_params = {**base_params, "tagged": tag, "page": page}

            data = await _fetch_stack_exchange_page(
                client,
                f"{app_config.eval_pipeline_config.stack_exchange.api_url}/questions",
                questions_params,
            )
            items: list[dict[str, Any]] = data.get("items", [])
            if not items:
                logger.info("No more questions available")
                break

            question_ids: list[int] = [q["question_id"] for q in items]
            answers_params = {**base_params}

            # Batch up to 100 question IDs via semicolons (avoids N+1).
            # e.g. .../questions/12345;67890;11121/answers
            answers_data = await _fetch_stack_exchange_page(
                client,
                f"{app_config.eval_pipeline_config.stack_exchange.api_url}/questions/{';'.join(map(str, question_ids))}/answers",
                answers_params,
            )

            # Build a map from question ID -> list of its answers
            answers_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for a in answers_data.get("items", []):
                answers_map[a["question_id"]].append(a)

            for item in items:
                if not item.get("is_answered"):
                    continue
                qid = item["question_id"]
                answers = answers_map.get(qid, [])
                # Skip questions that don't have an answer
                if not answers:
                    continue

                question = StackOverflowQuestionSchema(
                    question_id=qid,
                    title=item["title"],
                    link=item["link"],
                    body=item.get("body", ""),
                    body_markdown=item.get("body_markdown", ""),
                    score=item.get("score", 0),
                    answer_count=item.get("answer_count", 0),
                    view_count=item.get("view_count", 0),
                    creation_date=item.get("creation_date", 0),
                    tags=item.get("tags", []),
                    is_answered=True,
                    answer=_pick_best_answer(answers),
                )
                matched.append(question)
                if len(matched) >= num_issues:
                    break

            logger.info(
                "Collected %d/%d answered questions",
                len(matched),
                num_issues,
            )

            if not data.get("has_more"):
                logger.info("No more pages available")
                break

            await asyncio.sleep(
                app_config.eval_pipeline_config.stack_exchange.retry_sleep_secs
            )
            page += 1

    # Truncate to the number of issues requested
    matched = matched[:num_issues]
    output = Path(output_path)
    # Create the output directory if it does not exist
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write the matched discussions to a JSONL file
    await asyncio.to_thread(write_jsonl, output, matched)
    logger.info("Wrote %d questions to %s", len(matched), output)


# ===================================
# CLI app
# ===================================

app = typer.Typer(help="Fetch evaluation data", add_completion=False)


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Log the invoked command before running it."""
    logger.info("running fetch_eval_data command: %s", ctx.invoked_subcommand)


@app.command()
def github(
    url: str = app_config.eval_pipeline_config.defaults.github_url,
    num: int = app_config.eval_pipeline_config.defaults.num_issues,
    output: str = app_config.eval_pipeline_config.defaults.github_discussions_path,
    category: str = app_config.eval_pipeline_config.defaults.github_category,
) -> None:
    """Fetch answered and resolved discussions from a GitHub repo."""
    asyncio.run(afetch_data_from_github(url, num, output, category))


@app.command()
def stackoverflow(
    url: str = app_config.eval_pipeline_config.defaults.stackoverflow_url,
    num: int = app_config.eval_pipeline_config.defaults.num_issues,
    output: str = app_config.eval_pipeline_config.defaults.stackoverflow_questions_path,
    tag: str = app_config.eval_pipeline_config.defaults.stackoverflow_tag,
) -> None:
    """Fetch answered questions from Stack Overflow by tag."""
    asyncio.run(afetch_stack_exchange_data(url, num, output, tag))


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
