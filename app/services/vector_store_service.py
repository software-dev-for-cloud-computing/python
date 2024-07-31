import os
from typing import List

from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.vector_store import VectorStore
from app.models.objects.chunk_model import ChunkModel
from app.utils.logger import Logger

logger = Logger('app_logger')


class QdrantVectorStore(VectorStore):
    def __init__(self, embedding_model: EmbeddingModel):
        self.client = QdrantClient(url=os.getenv("VECTOR_STORE_URL"))
        self.embedding_model = embedding_model.get_model()

        # Initialize the collection if it does not exist
        if not self._get_collection(os.getenv("VECTOR_STORE_COLLECTION")):
            self._create_collection(collection_name=os.getenv("VECTOR_STORE_COLLECTION"))

        self.collection_connection = Qdrant(client=self.client,
                                            collection_name=os.getenv("VECTOR_STORE_COLLECTION"),
                                            embeddings=self.embedding_model,
                                            metadata_payload_key="metadata",
                                            )

    @logger.log_decorator(level="debug", message="Add chunks to collection")
    def add_chunks(self, chunks: List[ChunkModel], embedding_model: EmbeddingModel) -> None:
        documents = [Document(page_content=chunk.content, metadata=chunk.metadata) for chunk in chunks]
        self.collection_connection.from_documents(documents=documents,
                                                  embedding=self.embedding_model,
                                                  quantization_config=models.BinaryQuantization(
                                                      binary=models.BinaryQuantizationConfig(always_ram=False, ), ),
                                                  )

    def _get_collection(self, collection_name: str):
        return self.client.get_collection(collection_name=collection_name)

    @logger.log_decorator(level="debug", message="Create new collection")
    def _create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE, on_disk=True),
            hnsw_config=models.HnswConfigDiff(m=64, ef_construct=512, on_disk=True),
        )

    @logger.log_decorator(level="debug", message="Delete collection")
    def _delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name=collection_name)

    def _collection_exists(self, collection_name: str):
        return self.client.collection_exists(collection_name=collection_name)
