from abc import ABC, abstractmethod
from typing import List

from qdrant_client.http.models import CollectionInfo

from app.interfaces.embedding_model import EmbeddingModel
from app.models.objects.chunk_model import ChunkModel


class VectorStore(ABC):

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkModel], embedding_model: EmbeddingModel) -> None:
        pass

    @abstractmethod
    def _get_collection(self, collection_name: str) -> CollectionInfo:
        pass

    @abstractmethod
    def _create_collection(self, collection_name: str):
        pass

    @abstractmethod
    def _delete_collection(self, collection_name: str):
        pass

    @abstractmethod
    def _collection_exists(self, collection_name: str) -> bool:
        pass

