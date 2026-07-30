import pytest

from src.schemas.types import EnvironmentEnum, ErrorCodeEnum, RepoHandle


class TestEnvironmentEnum:
    def test_values(self) -> None:
        assert EnvironmentEnum.DEVELOPMENT.value == "development"
        assert EnvironmentEnum.PRODUCTION.value == "production"
        assert EnvironmentEnum.SANDBOX.value == "sandbox"

    def test_from_string(self) -> None:
        assert EnvironmentEnum("development") == EnvironmentEnum.DEVELOPMENT
        assert EnvironmentEnum("production") == EnvironmentEnum.PRODUCTION


class TestErrorCodeEnum:
    def test_values(self) -> None:
        assert ErrorCodeEnum.HTTP_ERROR.value == "http_error"
        assert ErrorCodeEnum.INTERNAL_SERVER_ERROR.value == "internal_server_error"
        assert ErrorCodeEnum.UNAUTHORIZED.value == "unauthorized"
        assert ErrorCodeEnum.UNEXPECTED_ERROR.value == "unexpected_error"


class TestRepoHandle:
    def test_is_named_tuple(self) -> None:
        handle = RepoHandle(owner="fastapi", name="fastapi")
        assert handle.owner == "fastapi"
        assert handle.name == "fastapi"

    def test_is_immutable(self) -> None:
        handle = RepoHandle(owner="fastapi", name="fastapi")
        with pytest.raises(AttributeError):
            # testing that NamedTuple fields are read-only
            handle.owner = "other"  # type: ignore

    def test_equality(self) -> None:
        a = RepoHandle(owner="a", name="b")
        b = RepoHandle(owner="a", name="b")
        c = RepoHandle(owner="a", name="c")
        assert a == b
        assert a != c
        assert a == ("a", "b")
