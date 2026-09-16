import msgspec

from src.api.core.response import MsgSpecJSONResponse


class TestMsgSpecJSONResponse:
    def test_none_renders_null_like_starlette(self) -> None:
        # Given / When
        response = MsgSpecJSONResponse(status_code=200, content=None)
        # Then
        assert response.body == b"null"

    def test_dict_round_trips_through_msgspec(self) -> None:
        # Given
        content = {"status": "error", "request_id": "abc", "path": "/ask"}
        # When
        response = MsgSpecJSONResponse(status_code=500, content=content)
        # Then
        assert msgspec.json.decode(response.body) == content
