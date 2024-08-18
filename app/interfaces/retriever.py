import os
from abc import abstractmethod, ABC
from typing import Optional

from app.interfaces.vector_store import VectorStore

import os
from dotenv import load_dotenv

load_dotenv()


class Retriever(ABC):
    vector_store: VectorStore

    @abstractmethod
    def get_document_retriever(self,
                               user_id: str,
                               document_id: Optional[str] = None,
                               conversation_id: Optional[str] = None,
                               k: int = int(os.getenv("MAX_K_RESULTS")),
                               ):
        pass

