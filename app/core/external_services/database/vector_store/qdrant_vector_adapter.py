import os

from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import CollectionInfo

from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.external_services.database.vector_store.vector_store_port import VectorStore
from app.core.utils.logger import Logger

logger = Logger('app_logger')

load_dotenv()

COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION")
MAX_K_RESULTS = os.getenv("MAX_K_RESULTS")


class VectorStoreQdrant(VectorStore):

    def __init__(self):
        self.collection_connection: Qdrant | None = None
        self.client = QdrantClient(url=os.getenv("VECTOR_STORE_URL"))

        # Initialize the collection if it does not exist
        if self.collection_exists(COLLECTION_NAME) is False:
            logger.log(level="warning", message="Creating collection")
            self.create_collection(collection_name=COLLECTION_NAME)

    def get_client(self) -> QdrantClient:
        return self.client

    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def get_connection(self, embedding_model: EmbeddingModel) -> Qdrant:
        if self.collection_connection is None:
            self.collection_connection = Qdrant(client=self.client,
                                                collection_name=COLLECTION_NAME,
                                                embeddings=embedding_model.get_model(),
                                                metadata_payload_key="metadata",
                                                )

        return self.collection_connection


    def get_collection(self, collection_name: str) -> CollectionInfo | None:
        if self.client.collection_exists(collection_name=collection_name):
            return self.client.get_collection(collection_name=collection_name)
        return None

    def get_collections(self):
        return self.client.get_collections()

    @logger.log_decorator(level="debug", message="Create new collection")
    def create_collection(self, collection_name: str):
        if not self.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE, on_disk=True),
                hnsw_config=models.HnswConfigDiff(m=64, ef_construct=512, on_disk=True),
            )

    @logger.log_decorator(level="debug", message="Delete collection")
    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name=collection_name)
