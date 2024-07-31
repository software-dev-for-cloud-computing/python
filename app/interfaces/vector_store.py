from abc import ABC, abstractmethod
from typing import List

from app.interfaces.embedding_model import EmbeddingModel
from app.models.objects.chunk_model import ChunkModel


class VectorStore(ABC):

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkModel], embedding_model: EmbeddingModel) -> None:
        pass

