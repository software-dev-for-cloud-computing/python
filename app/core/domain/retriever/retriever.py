import os
from abc import abstractmethod, ABC
from typing import Optional
from app.core.external_services.database.vector_store.vector_store_port import VectorStore
import os
from dotenv import load_dotenv

load_dotenv()


class Retriever(ABC):
    vector_store: VectorStore

    @abstractmethod
    def _filter_conditions(self,
                           user_id: str,
                           document_id: Optional[str] = None,
                           conversation_id: Optional[str] = None):
        pass

    @abstractmethod
    def get_mmr_retriever(self,
                          user_id: str,
                          document_id: Optional[str] = None,
                          conversation_id: Optional[str] = None,
                          k: int = int(os.getenv("MAX_K_RESULTS")),
                          ):
        pass

    @abstractmethod
    def get_similarity_retriever(self,
                                 user_id: str,
                                 document_id: Optional[str] = None,
                                 conversation_id: Optional[str] = None,
                                 k: int = int(os.getenv("MAX_K_RESULTS")),
                                 ):
        pass
