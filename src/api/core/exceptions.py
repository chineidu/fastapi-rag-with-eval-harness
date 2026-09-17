"""API error taxonomy and exception handlers with a uniform envelope."""

import logging

from fastapi import Request, status
from fastapi.exceptions import HTTPException, RequestValidationError

from src import create_logger
from src.api.core.response import MsgSpecJSONResponse
from src.schemas.types import ErrorCodeEnum

logger = create_logger(name=__name__)

__all__ = [
    "BaseAPIError",
    "GenerationError",
    "RequestTimeoutError",
    "UpstreamUnavailableError",
    "aapi_error_handler",
    "ahttp_error_handler",
    "arequest_validation_handler",
    "aunhandled_exception_handler",
    "is_upstream_connection_error",
    "map_provider_error",
]


class BaseAPIError(Exception):
    """Base exception for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: ErrorCodeEnum = ErrorCodeEnum.INTERNAL_SERVER_ERROR,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store the message, status code, error code, and headers."""
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.headers = headers


class GenerationError(BaseAPIError):
    """Exception raised when retrieval-augmented generation fails."""

    def __init__(self, details: str) -> None:
        """Build a 500 generation error from provider details."""
        message = f"Generation error: {details}"
        super().__init__(
            message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCodeEnum.GENERATION_ERROR,
        )


class RequestTimeoutError(BaseAPIError):
    """Exception raised when a request exceeds the configured timeout."""

    def __init__(self, details: str) -> None:
        """Build a 504 timeout error from timeout details."""
        message = f"Request timeout: {details}"
        super().__init__(
            message,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code=ErrorCodeEnum.TIMEOUT_ERROR,
        )


class UpstreamUnavailableError(BaseAPIError):
    """Exception raised when the LLM provider is unreachable."""

    def __init__(self, details: str) -> None:
        """Build a 503 upstream error from provider details."""
        message = f"Upstream unavailable: {details}"
        super().__init__(
            message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ErrorCodeEnum.UPSTREAM_UNAVAILABLE,
        )


def is_upstream_connection_error(exc: Exception) -> bool:
    """Check whether an error chain signals a provider connection failure.

    Parameters
    ----------
    exc : Exception
        The raised error, possibly wrapping provider SDK errors.

    Returns
    -------
    bool
        True when any link in the cause chain names a known
        connection failure, without importing LLM libraries.

    """
    # Match by class name to avoid coupling the API layer to LLM SDKs.
    names: set[str] = {
        "APIConnectionError",
        "ConnectError",
        "InstructorRetryException",
    }
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in names:
            return True
        current = current.__cause__ or current.__context__
    try:
        message = str(exc)
    except Exception:  # noqa: BLE001 - str() must never break the bool contract
        return False
    return "nodename nor servname" in message


def map_provider_error(
    exc: Exception,
    *,
    logger: logging.Logger,
    operation: str,
    query_preview: str,
) -> BaseAPIError:
    """Map a generation failure to 503 upstream or sanitized 500.

    Parameters
    ----------
    exc : Exception
        The raised error, possibly wrapping provider SDK errors.
    logger : logging.Logger
        Caller logger, so the log source stays in the route module.
    operation : str
        Route label for the log line (for example ``Ask``).
    query_preview : str
        Truncated query for the log line.

    Returns
    -------
    BaseAPIError
        Upstream error for known connection failures, else a
        sanitized generation error.

    """
    # Map connection failures to 503 and sanitise unknowns to 500.
    if is_upstream_connection_error(exc):
        logger.warning("%s upstream unreachable for query %r", operation, query_preview)
        return UpstreamUnavailableError("LLM provider unreachable, try again later")
    logger.exception("%s failed for query %r", operation, query_preview)
    return GenerationError("Generation failed, try again later")


def _error_content(request: Request, message: str, code: str) -> dict:
    """Build the uniform error envelope for a request."""
    return {
        "status": "error",
        "error": {"message": message, "code": code},
        "request_id": getattr(request.state, "request_id", "N/A"),
        "path": request.url.path,
    }


async def aapi_error_handler(
    request: Request, exc: BaseAPIError
) -> MsgSpecJSONResponse:
    """Handle custom API errors with their declared status code.

    Parameters
    ----------
    request : Request
        The incoming request object.
    exc : BaseAPIError
        The exception that was raised.

    Returns
    -------
    MsgSpecJSONResponse
        A JSON response with error details and the exception status code.

    """
    return MsgSpecJSONResponse(
        status_code=exc.status_code,
        content=_error_content(request, exc.message, exc.error_code),
        headers=exc.headers,
    )


async def arequest_validation_handler(
    request: Request, exc: RequestValidationError
) -> MsgSpecJSONResponse:
    """Handle request validation failures as 422 invalid-input errors.

    Parameters
    ----------
    request : Request
        The incoming request object.
    exc : RequestValidationError
        The validation error raised by FastAPI.

    Returns
    -------
    MsgSpecJSONResponse
        A JSON response joining every field error into one message.

    """
    details = "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
    )
    if not details:
        details = "Invalid request."
    return MsgSpecJSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_content(request, details, ErrorCodeEnum.INVALID_INPUT),
    )


async def ahttp_error_handler(
    request: Request, exc: HTTPException
) -> MsgSpecJSONResponse:
    """Handle HTTP exceptions raised in the application.

    Parameters
    ----------
    request : Request
        The incoming request object.
    exc : HTTPException
        The HTTP exception that was raised.

    Returns
    -------
    MsgSpecJSONResponse
        A JSON response with the exception detail and status code.

    """
    # HTTPException detail may be None; never leak a "None" message.
    message = exc.detail if exc.detail is not None else "HTTP error"
    return MsgSpecJSONResponse(
        status_code=exc.status_code,
        content=_error_content(request, str(message), ErrorCodeEnum.HTTP_ERROR),
        headers=exc.headers,
    )


async def aunhandled_exception_handler(
    request: Request, exc: Exception
) -> MsgSpecJSONResponse:
    """Handle uncaught exceptions as internal server errors.

    Parameters
    ----------
    request : Request
        The incoming request object.
    exc : Exception
        The uncaught exception.

    Returns
    -------
    MsgSpecJSONResponse
        A JSON response hiding internal details behind a generic message.

    """
    logger.exception("Unhandled exception at app layer: %s", exc)
    return MsgSpecJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_content(
            request,
            "An unexpected server error occurred.",
            ErrorCodeEnum.UNEXPECTED_ERROR,
        ),
    )
