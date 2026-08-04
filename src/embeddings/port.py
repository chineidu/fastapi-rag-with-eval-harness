from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderPort(Protocol):
    """Contract for text-embedding implementations.

    Implementations must expose the active model id, the embedding
    dimension, and a batched ``embed_texts`` method. The harness and the
    labeling pipeline only ever depend on this interface, so the local and
    API embedders are interchangeable.
    """

    model_id: str

    @property
    def dim(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
