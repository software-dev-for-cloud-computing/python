import os
from abc import ABC, abstractmethod
from typing import List, Type, Any

from pydantic import PositiveInt, Field

from app.models.objects.chunk_model import ChunkModel

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL")


class EmbeddingModel(ABC):
    dimensions: PositiveInt = Field(1024, ge=124, le=4096)
    model_name: str = DEFAULT_MODEL
    api_key: str = Field(min_length=30)
    tokenizer: Any

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        pass

    @abstractmethod
    def set_dimensions(self, dimensions: PositiveInt) -> None:
        pass

    @abstractmethod
    def estimate_tokens_text(self, text: str) -> int:
        pass

    @abstractmethod
    def estimate_tokens_chunks(self, chunks: List[ChunkModel]) -> int:
        pass

