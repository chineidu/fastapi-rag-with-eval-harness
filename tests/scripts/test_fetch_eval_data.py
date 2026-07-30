from unittest import mock

import httpx
import pytest

from src.schemas.types import RepoHandle


class TestParseRepoUrl:
    def test_extracts_owner_and_repo(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        result = mod._parse_repo_url("https://github.com/fastapi/fastapi")
        assert result == RepoHandle(owner="fastapi", name="fastapi")

    def test_handles_trailing_slash(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        result = mod._parse_repo_url("https://github.com/owner/repo/")
        assert result == RepoHandle(owner="owner", name="repo")

    def test_handles_git_suffix(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        result = mod._parse_repo_url("https://github.com/owner/repo.git")
        assert result == RepoHandle(owner="owner", name="repo")

    def test_raises_on_invalid_url(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        with pytest.raises(ValueError, match="Expected GitHub repo URL"):
            mod._parse_repo_url("https://github.com")


class TestParseStackExchangeUrl:
    def test_extracts_site_from_hostname(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        assert (
            mod._parse_stack_exchange_url("https://stackoverflow.com/questions")
            == "stackoverflow"
        )

    def test_handles_no_hostname(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        result = mod._parse_stack_exchange_url("just-a-string")
        assert result == "just-a-string"

    def test_handles_subdomain(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        result = mod._parse_stack_exchange_url("https://meta.stackexchange.com")
        assert result == "meta"


class TestPickBestAnswer:
    def test_returns_accepted_answer(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        answers = [
            {
                "answer_id": 1,
                "question_id": 1,
                "score": 5,
                "is_accepted": False,
                "body": "a",
                "creation_date": 1000,
            },
            {
                "answer_id": 2,
                "question_id": 1,
                "score": 3,
                "is_accepted": True,
                "body": "b",
                "creation_date": 1000,
            },
            {
                "answer_id": 3,
                "question_id": 1,
                "score": 10,
                "is_accepted": False,
                "body": "c",
                "creation_date": 1000,
            },
        ]
        result = mod._pick_best_answer(answers)
        assert result.answer_id == 2

    def test_returns_highest_scored_when_none_accepted(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        answers = [
            {
                "answer_id": 1,
                "question_id": 1,
                "score": 5,
                "is_accepted": False,
                "body": "a",
                "creation_date": 1000,
            },
            {
                "answer_id": 2,
                "question_id": 1,
                "score": 10,
                "is_accepted": False,
                "body": "b",
                "creation_date": 1000,
            },
            {
                "answer_id": 3,
                "question_id": 1,
                "score": 3,
                "is_accepted": False,
                "body": "c",
                "creation_date": 1000,
            },
        ]
        result = mod._pick_best_answer(answers)
        assert result.answer_id == 2

    def test_injects_link_when_missing(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        answers = [
            {
                "answer_id": 42,
                "question_id": 1,
                "score": 1,
                "is_accepted": False,
                "body": "a",
                "creation_date": 1000,
            },
        ]
        result = mod._pick_best_answer(answers)
        assert result.link == "https://stackoverflow.com/a/42"


class TestIsAnsweredAndResolved:
    def test_true_when_answered_and_resolved(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        from src.schemas.output import DiscussionNodeSchema

        node = DiscussionNodeSchema.model_validate(
            {
                "id": "x",
                "number": 1,
                "title": "t",
                "url": "u",
                "body": "b",
                "bodyText": "b",
                "createdAt": "2024-01-01T00:00:00Z",
                "closedAt": None,
                "answerChosenAt": None,
                "upvoteCount": 1,
                "isAnswered": True,
                "stateReason": "RESOLVED",
                "category": {"name": "Q", "slug": "q"},
                "labels": {"nodes": []},
                "comments": {"totalCount": 0},
                "answer": None,
            }
        )
        assert mod._is_answered_and_resolved(node) is True

    def test_false_when_not_answered(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        from src.schemas.output import DiscussionNodeSchema

        node = DiscussionNodeSchema.model_validate(
            {
                "id": "x",
                "number": 1,
                "title": "t",
                "url": "u",
                "body": "b",
                "bodyText": "b",
                "createdAt": "2024-01-01T00:00:00Z",
                "closedAt": None,
                "answerChosenAt": None,
                "upvoteCount": 1,
                "isAnswered": False,
                "stateReason": "RESOLVED",
                "category": {"name": "Q", "slug": "q"},
                "labels": {"nodes": []},
                "comments": {"totalCount": 0},
                "answer": None,
            }
        )
        assert mod._is_answered_and_resolved(node) is False

    def test_false_when_not_resolved(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        from src.schemas.output import DiscussionNodeSchema

        node = DiscussionNodeSchema.model_validate(
            {
                "id": "x",
                "number": 1,
                "title": "t",
                "url": "u",
                "body": "b",
                "bodyText": "b",
                "createdAt": "2024-01-01T00:00:00Z",
                "closedAt": None,
                "answerChosenAt": None,
                "upvoteCount": 1,
                "isAnswered": True,
                "stateReason": "DUPLICATE",
                "category": {"name": "Q", "slug": "q"},
                "labels": {"nodes": []},
                "comments": {"totalCount": 0},
                "answer": None,
            }
        )
        assert mod._is_answered_and_resolved(node) is False


class TestRunQuery:
    async def test_200_ok(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": {"ok": True}})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mod._run_query(client, query="q", variables={})
            assert result == {"data": {"ok": True}}

    async def test_non_200_raises(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "internal"})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await mod._run_query(client, query="q", variables={})

    async def test_429_retries_then_succeeds(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        call_count = 0

        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json={"data": {"ok": True}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mod._run_query(client, query="q", variables={})
            assert result == {"data": {"ok": True}}
            assert call_count == 2

    async def test_max_retries_exceeded(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"retry-after": "0"})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="Max retries exceeded"):
                await mod._run_query(client, query="q", variables={})

    async def test_graphql_errors_raises(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"errors": [{"message": "bad"}]})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="GraphQL errors"):
                await mod._run_query(client, query="q", variables={})


class TestResolveCategoryId:
    async def test_found(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": {
                        "repository": {
                            "discussionCategories": {
                                "nodes": [
                                    {"id": "cat-1", "slug": "announcements"},
                                    {"id": "cat-2", "slug": "questions"},
                                ]
                            }
                        }
                    }
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await mod._resolve_category_id(client, "o", "r", "questions")
            assert result == "cat-2"

    async def test_not_found_raises(
        self, fetch_eval_data, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = fetch_eval_data
        monkeypatch.setattr(
            mod, "_get_github_auth_headers", lambda: {"Authorization": "Bearer test"}
        )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": {
                        "repository": {
                            "discussionCategories": {
                                "nodes": [
                                    {"id": "cat-1", "slug": "announcements"},
                                ]
                            }
                        }
                    }
                },
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="not found"):
                await mod._resolve_category_id(client, "o", "r", "questions")


class TestFetchStackExchangePage:
    async def test_200_ok(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        # Patch asyncio.sleep to avoid real delays
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"items": []})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with mock.patch("asyncio.sleep", return_value=None):
                result = await mod._fetch_stack_exchange_page(client, "http://test", {})
                assert result == {"items": []}

    async def test_error_id_raises(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"error_id": 404, "error_message": "not found"}
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="Stack Exchange API error"):
                await mod._fetch_stack_exchange_page(client, "http://test", {})

    async def test_backoff_respected(self, fetch_eval_data) -> None:
        mod = fetch_eval_data
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"items": [], "backoff": 5})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with mock.patch("asyncio.sleep") as mock_sleep:
                _ = await mod._fetch_stack_exchange_page(client, "http://test", {})
                mock_sleep.assert_called_once_with(5)
