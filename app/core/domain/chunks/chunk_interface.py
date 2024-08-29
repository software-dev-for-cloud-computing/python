from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.domain.chunks.chunk_model import ChunkModel


class ChunkInterface(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[ChunkModel], embedding_model: EmbeddingModel) -> None:
        pass

    @abstractmethod
    def get_chunks(self, user_id: str, document_id: str) -> List[ChunkModel]:
        pass

    @abstractmethod
    def get_all_chunks(self, user_id):
        pass

    @abstractmethod
    def delete_chunks(self, user_id: str, document_id: Optional[str] = None):
        pass

    @abstractmethod
    def search_chunks(self,
                      embedding_model: EmbeddingModel,
                      query: str,
                      user_id: str,
                      document_id: Optional[str] = None,
                      k: int = 5):
        pass