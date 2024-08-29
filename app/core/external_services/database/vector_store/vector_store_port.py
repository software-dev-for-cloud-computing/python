from abc import ABC, abstractmethod
from typing import Any

from qdrant_client.http.models import CollectionInfo

from app.core.external_services.embedding.embedding_port import EmbeddingModel


class VectorStore(ABC):

    @abstractmethod
    def get_client(self) -> Any:
        pass

    @abstractmethod
    def get_connection(self, embedding_model: EmbeddingModel) -> Any:
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

