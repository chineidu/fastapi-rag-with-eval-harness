import json

from fastapi import Request, status
from fastapi.exceptions import HTTPException, RequestValidationError

from src.api.core.exceptions import (
    BaseAPIError,
    GenerationError,
    RequestTimeoutError,
    aapi_error_handler,
    ahttp_error_handler,
    arequest_validation_handler,
    aunhandled_exception_handler,
)
from src.schemas.types import ErrorCodeEnum


def make_request(path: str = "/ask", request_id: str | None = "test-id") -> Request:
    """Build a minimal Starlette request, optionally stamped with an ID."""
    scope: dict = {"type": "http", "method": "POST", "path": path, "headers": []}
    if request_id is not None:
        scope["state"] = {"request_id": request_id}
    return Request(scope)


def decode_body(response) -> dict:
    """Decode a rendered JSON response body."""
    return json.loads(response.body.decode())


class TestBaseAPIError:
    def test_defaults(self) -> None:
        # Given / When
        exc = BaseAPIError("boom")
        # Then
        assert exc.message == "boom"
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.error_code == ErrorCodeEnum.INTERNAL_SERVER_ERROR
        assert exc.headers is None

    def test_forwards_headers(self) -> None:
        # Given / When
        exc = BaseAPIError("boom", headers={"retry-after": "1"})
        # Then
        assert exc.headers == {"retry-after": "1"}

    def test_generation_error(self) -> None:
        # Given / When
        exc = GenerationError("provider down")
        # Then
        assert exc.message == "Generation error: provider down"
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.error_code == ErrorCodeEnum.GENERATION_ERROR

    def test_request_timeout_error(self) -> None:
        # Given / When
        exc = RequestTimeoutError("slow provider")
        # Then
        assert exc.message == "Request timeout: slow provider"
        assert exc.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        assert exc.error_code == ErrorCodeEnum.TIMEOUT_ERROR


class TestAapiErrorHandler:
    async def test_envelope_and_status(self) -> None:
        # Given
        request = make_request()
        # When
        response = await aapi_error_handler(request, GenerationError("nope"))
        # Then
        assert response.status_code == 500
        assert decode_body(response) == {
            "status": "error",
            "error": {
                "message": "Generation error: nope",
                "code": "generation_error",
            },
            "request_id": "test-id",
            "path": "/ask",
        }

    async def test_forwards_custom_headers(self) -> None:
        # Given
        request = make_request()
        exc = BaseAPIError("wait", status_code=429, headers={"retry-after": "1"})
        # When
        response = await aapi_error_handler(request, exc)
        # Then
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"

    async def test_request_id_falls_back_without_middleware(self) -> None:
        # Given
        request = make_request(request_id=None)
        # When
        response = await aapi_error_handler(request, BaseAPIError("boom"))
        # Then
        assert decode_body(response)["request_id"] == "N/A"


class TestRequestValidationHandler:
    async def test_joins_field_errors(self) -> None:
        # Given
        request = make_request()
        exc = RequestValidationError(
            errors=[
                {"loc": ("body", "query"), "msg": "field required", "type": "missing"},
                {
                    "loc": ("body", "top_k"),
                    "msg": "out of range",
                    "type": "value_error",
                },
            ]
        )
        # When
        response = await arequest_validation_handler(request, exc)
        # Then
        assert response.status_code == 422
        body = decode_body(response)
        assert body["error"]["code"] == "invalid_input"
        assert "body.query: field required" in body["error"]["message"]
        assert "body.top_k: out of range" in body["error"]["message"]

    async def test_empty_errors_falls_back_to_generic_message(self) -> None:
        # Given
        request = make_request()
        exc = RequestValidationError(errors=[])
        # When
        response = await arequest_validation_handler(request, exc)
        # Then
        assert decode_body(response)["error"]["message"] == "Invalid request."


class TestHttpErrorHandler:
    async def test_uses_detail_and_status(self) -> None:
        # Given
        request = make_request()
        # When
        response = await ahttp_error_handler(
            request, HTTPException(status_code=404, detail="missing")
        )
        # Then
        assert response.status_code == 404
        body = decode_body(response)
        assert body["error"] == {"message": "missing", "code": "http_error"}

    async def test_none_detail_never_renders_none_string(self) -> None:
        # Given: HTTPException fills a missing detail with the status phrase.
        request = make_request()
        # When
        response = await ahttp_error_handler(
            request, HTTPException(status_code=403, detail=None)
        )
        # Then
        assert decode_body(response)["error"]["message"] == "Forbidden"


class TestUnhandledExceptionHandler:
    async def test_hides_internal_details(self) -> None:
        # Given
        request = make_request()
        # When
        response = await aunhandled_exception_handler(request, RuntimeError("secret"))
        # Then
        assert response.status_code == 500
        body = decode_body(response)
        assert body["error"] == {
            "message": "An unexpected server error occurred.",
            "code": "unexpected_error",
        }
        assert "secret" not in json.dumps(body)
