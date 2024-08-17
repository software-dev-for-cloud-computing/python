from abc import ABC, abstractmethod
from typing import List, Any, Optional

from langchain_qdrant import Qdrant
from qdrant_client.http.models import CollectionInfo

from app.interfaces.embedding_model import EmbeddingModel
from app.models.objects.chunk_model import ChunkModel


class VectorStore(ABC):

    @abstractmethod
    def get_connection(self, embedding_model: EmbeddingModel) -> Any:
        pass

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
    def delete_chunks(self, user_id: str, document_id: str):
        pass

    @abstractmethod
    def search_chunks(self, embedding_model: EmbeddingModel, query: str, user_id: str, document_id: Optional[str] = None, k: int = 5):
        pass

    @abstractmethod
    def get_collection(self, collection_name: str) -> CollectionInfo:
        pass

    @abstractmethod
    def create_collection(self, collection_name: str):
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        pass

