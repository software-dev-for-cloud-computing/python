from abc import abstractmethod, ABC
from typing import Optional

from app.interfaces.vector_store import VectorStore


class Retriever(ABC):

    vector_store: VectorStore

    @abstractmethod
    def get_document_retriever(self, user_id: str, k: int, document_id: Optional[str] = None):
        pass

    @abstractmethod
    def search(self, query: str, user_id: str, document_id: str, k: int):
        pass

