from abc import abstractmethod, ABC

from app.interfaces.vector_store import VectorStore


class Retriever(ABC):

    vector_store: VectorStore

    @abstractmethod
    def get_document_retriever_for_user_id(self, user_id: str):
        pass

    @abstractmethod
    def get_document_retriever_for_user_id_and_document_id(self, user_id: str, document_id: str):
        pass

    @abstractmethod
    def search(self, query: str, user_id: str, k: int):
        pass

    @abstractmethod
    def search_within_document(self, query: str, user_id: str, document_id: str, k: int):
        pass
