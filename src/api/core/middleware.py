"""Request ID middleware."""

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

__all__ = ["REQUEST_ID_HEADER", "REQUEST_ID_STATE_KEY", "RequestIDMiddleware"]

REQUEST_ID_HEADER = "x-request-id"
REQUEST_ID_STATE_KEY = "request_id"

Scope = Any
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class RequestIDMiddleware:
    """Stamp each HTTP request with an ID for the error envelope.

    Reuse an incoming ``x-request-id`` header when present, otherwise
    mint a random hex ID. The ID lands on ``request.state.request_id``
    (read by the error envelope) and is echoed back as a response header.
    """

    def __init__(self, app: Any) -> None:
        """Store the downstream ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Attach the request ID and echo it on the response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Prefer the caller-supplied ID so retries stay correlatable.
        headers = dict(scope.get("headers", []))
        request_id = headers.get(REQUEST_ID_HEADER.encode(), b"").decode().strip()
        if not request_id:
            request_id = uuid4().hex
        # Link into request.state via the shared scope state mapping.
        state = scope.get("state")
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state[REQUEST_ID_STATE_KEY] = request_id

        async def sender(message: Message) -> None:
            # Echo the ID on the response start message only.
            if message["type"] == "http.response.start":
                headers_out = list(message.get("headers", []))
                headers_out.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers_out
            await send(message)

        await self.app(scope, receive, sender)
