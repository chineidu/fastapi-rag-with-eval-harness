import asyncio
import json
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from src import create_logger
from src.config import app_settings
from src.schemas.output import GitHubAPIResponseSchema
from src.schemas.types import RepoHandle

logger = create_logger(name=__name__)

GITHUB_GRAPHQL_URL: str = "https://api.github.com/graphql"
PAGE_SIZE: int = 100
MAX_RETRIES: int = 3
RETRY_SLEEP_SECS: float = 0.5


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
    # token = os.environ.get("GITHUB_TOKEN")
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
    headers = _get_github_auth_headers()
    for attempt in range(MAX_RETRIES):
        response = await client.post(
            GITHUB_GRAPHQL_URL,
            headers=headers,
            json={"query": query, "variables": variables},
        )
        if response.status_code == 429:
            retry_after = int(response.headers.get("retry-after", 10))
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %ds",
                attempt + 1,
                MAX_RETRIES,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            continue
        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned HTTP {response.status_code}: {response.text}"
            )
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        return data
    raise RuntimeError("Max retries exceeded for GitHub API rate limiting")


async def _resolve_category_id(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    slug: str,
) -> str:
    """Resolve discussion category ID from slug."""
    data = await _run_query(
        client, query=CATEGORY_ID_QUERY, variables={"owner": owner, "name": repo}
    )
    categories = data["data"]["repository"]["discussionCategories"]["nodes"]
    for cat in categories:
        if cat["slug"] == slug:
            return cat["id"]

    # If the requested category slug is not found, inform the user
    available = [c["slug"] for c in categories]
    raise RuntimeError(
        f"Category '{slug}' not found in {owner}/{repo}. Available: {available}"
    )


def _is_answered_and_resolved(node: GitHubAPIResponseSchema) -> bool:
    """Check if a discussion is answered and resolved."""
    return bool(node.is_answered) and node.state_reason == "RESOLVED"


def _write_jsonl(path: Path, records: list[GitHubAPIResponseSchema]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                f"{json.dumps(record.model_dump(by_alias=True), ensure_ascii=False, default=str)}\n"
            )


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
        category_id = await _resolve_category_id(client, owner, repo, category_slug)
        logger.info("Resolved category ID for '%s'", category_slug)

        matched: list[GitHubAPIResponseSchema] = []
        cursor: str | None = None

        while len(matched) < num_issues:
            data = await _run_query(
                client,
                query=DISCUSSIONS_QUERY,
                variables={
                    "owner": owner,
                    "name": repo,
                    "cursor": cursor,
                    "category_id": category_id,
                    "first": PAGE_SIZE,
                },
            )
            page = data["data"]["repository"]["discussions"]
            parsed = (GitHubAPIResponseSchema.model_validate(n) for n in page["nodes"])
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
            await asyncio.sleep(RETRY_SLEEP_SECS)

    matched = matched[:num_issues]
    matched.sort(key=lambda n: n.upvote_count, reverse=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_write_jsonl, output, matched)
    logger.info("Wrote %d discussions to %s", len(matched), output)


if __name__ == "__main__":
    url: str = "https://github.com/fastapi/fastapi"
    num_issues: int = 5
    output_path: str = "data/fastapi_discussions.jsonl"
    asyncio.run(afetch_data_from_github(url, num_issues, output_path))
