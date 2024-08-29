import os
from abc import ABC, abstractmethod
from typing import List, Any

from dotenv import load_dotenv
from pydantic import PositiveInt, Field

from app.core.domain.chunks.chunk_model import ChunkModel

load_dotenv()


#DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL")


class EmbeddingModel(ABC):
    dimensions: PositiveInt = Field(1024, ge=124, le=4096)
    model_name: str  # = DEFAULT_MODEL
    api_key: str = Field(min_length=30)
    tokenizer: Any

    @abstractmethod
    def __init__(self,
                 api_key: str,
                 model_name: str,
                 dimensions: PositiveInt):
        pass

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
