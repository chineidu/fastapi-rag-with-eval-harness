from typing import Any

import msgspec
from starlette.requests import Request

from src.api.core.middleware import (
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
)


async def echo_app(scope: dict, receive: Any, send: Any) -> None:
    """Downstream app echoing the stamped request ID as JSON."""
    request = Request(scope)
    body = msgspec.json.encode({"request_id": request.state.request_id})
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def run_middleware(
    middleware: RequestIDMiddleware, scope: dict
) -> tuple[dict, dict]:
    """Drive the middleware with a raw ASGI exchange, return start + body."""
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b""}

    async def send(message: dict) -> None:
        messages.append(message)

    await middleware(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = next(m for m in messages if m["type"] == "http.response.body")
    return start, body


def http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    """Build a minimal HTTP scope."""
    return {"type": "http", "method": "GET", "path": "/", "headers": headers or []}


async def _inert_receive() -> dict:
    """Inert receive channel for non-HTTP scopes."""
    raise AssertionError("lifespan probe must not receive")


async def _inert_send(message: dict) -> None:
    """Inert send channel for non-HTTP scopes."""
    raise AssertionError(f"lifespan probe must not send: {message!r}")


class TestRequestIDMiddleware:
    async def test_mints_id_and_echoes_header(self) -> None:
        # Given
        scope = http_scope()
        # When
        start, body = await run_middleware(RequestIDMiddleware(echo_app), scope)
        # Then
        echoed = dict(start["headers"])[REQUEST_ID_HEADER.encode()].decode()
        assert len(echoed) == 32
        assert msgspec.json.decode(body["body"]) == {"request_id": echoed}
        assert scope["state"]["request_id"] == echoed

    async def test_reuses_incoming_id(self) -> None:
        # Given
        scope = http_scope(headers=[(REQUEST_ID_HEADER.encode(), b"caller-123")])
        # When
        start, body = await run_middleware(RequestIDMiddleware(echo_app), scope)
        # Then
        assert dict(start["headers"])[REQUEST_ID_HEADER.encode()] == b"caller-123"
        assert msgspec.json.decode(body["body"]) == {"request_id": "caller-123"}

    async def test_non_http_scope_passes_through(self) -> None:
        # Given
        seen: list[dict] = []

        async def spy(scope: dict, receive: Any, send: Any) -> None:
            seen.append(scope)

        scope = {"type": "lifespan"}
        # When: lifespan probes carry no channels, so use inert stubs.
        await RequestIDMiddleware(spy)(scope, _inert_receive, _inert_send)
        # Then
        assert seen == [scope]
        assert "state" not in scope
