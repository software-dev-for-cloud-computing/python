from langchain_openai.embeddings import OpenAIEmbeddings

# Langchain Qdrant imports
from qdrant_client import QdrantClient as QdrantClientBase, models, QdrantClient
from langchain_qdrant import Qdrant

from core.config import QDRANT_HOST, OPENAI_API_KEY
from services.rag.openai_models import OpenAIModels
from utils.singleton import SingletonMeta


def _create_client():
    return QdrantClient(url=QDRANT_HOST)


class VectorStoreService(metaclass=SingletonMeta):
    def __init__(self):
        self.qdrant_client = _create_client()
        self.embeddings = OpenAIModels(OPENAI_API_KEY).get_embedding()
        self.vector_store = None
        self.url = None
        self.collection_name = None

    def connect(self, collection_name: str):
        if self.vector_store is None:
            self.collection_name = collection_name
            self.vector_store = Qdrant(self.qdrant_client,
                                       collection_name=collection_name,
                                       embeddings=self.embeddings)
        return self.vector_store

    def get_vectorstore(self):
        return self.vector_store

    def get_client(self) -> QdrantClient:
        return self.qdrant_client
