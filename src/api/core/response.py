"""JSON responses rendered with msgspec."""

from typing import Any

import msgspec
from fastapi.responses import JSONResponse

__all__ = ["MsgSpecJSONResponse"]


class MsgSpecJSONResponse(JSONResponse):
    """JSON response using msgspec for faster rendering."""

    def render(self, content: Any) -> bytes:
        """Render the content to JSON bytes using msgspec.

        Parameters
        ----------
        content : Any
            The content to be rendered as JSON.

        Returns
        -------
        bytes
            The JSON-encoded bytes of the content.

        """
        # Match Starlette: None renders as null instead of raising.
        if content is None:
            return b"null"
        return msgspec.json.encode(content)
