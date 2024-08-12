import os
from typing import List, Collection, Optional

from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError, HTTPException
from langchain_core.documents import Document
from langchain_qdrant import Qdrant, QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import CollectionInfo

from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.vector_store import VectorStore
from app.models.objects.chunk_model import ChunkModel
from app.utils.logger import Logger

logger = Logger('app_logger')

load_dotenv()

COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION")


class VectorStoreQdrant(VectorStore):

    def __init__(self):
        self.collection_connection: Qdrant | None = None
        self.client = QdrantClient(url=os.getenv("VECTOR_STORE_URL"))

        # Initialize the collection if it does not exist
        if self.collection_exists(COLLECTION_NAME) is False:
            logger.log(level="warning", message="Creating collection")
            self.create_collection(collection_name=COLLECTION_NAME)

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

    @logger.log_decorator(level="debug", message="Add chunks to collection")
    def add_chunks(self, chunks: List[ChunkModel], embedding_model: EmbeddingModel) -> None:

        documents = [Document(page_content=chunk.content, metadata=chunk.metadata) for chunk in chunks]

        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embedding_model.get_model(),
            url=os.getenv("VECTOR_STORE_URL"),
            collection_name=os.getenv("VECTOR_STORE_COLLECTION"),
            metadata_payload_key="metadata",
        )

    @logger.log_decorator(level="debug", message="Get chunks to collection")
    def get_chunks(self, user_id: str, document_id: str):  # -> List[ChunkModel]:
        try:
            return self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.owner_id",
                            match=models.MatchValue(value=user_id)
                        ),
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )

        except RequestValidationError:
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_all_chunks(self, user_id: str):
        return self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.owner_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
        )

    @logger.log_decorator(level="debug", message="Delete chunks from one doucment from the collection")
    def delete_chunks(self, document_id: str, user_id: str):
        try:
            return self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.document_id",
                                match=models.MatchValue(value=document_id),
                            ),
                        ],
                    )
                ),
            )
            # return SuccessResponse(message="Documents deleted successfully")
        except RequestValidationError:
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @logger.log_decorator(level="debug", message="Search for chunks")
    def search_chunks(self, embedding_model: EmbeddingModel, query: str, user_id: str, document_id: Optional[str] = None, k: int = 5):
        try:
            connection = self.get_connection(embedding_model)

            if document_id is None:
                filter = {"owner_id": user_id}
            else:
                filter = {"owner_id": user_id, "document_id": document_id}

            return connection.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
            )

        except RequestValidationError:
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    def get_collection(self, collection_name: str) -> CollectionInfo | None:
        if self.client.collection_exists(collection_name=collection_name):
            return self.client.get_collection(collection_name=collection_name)
        return None

    def get_collections(self):  # -> List[CollectionInfo]:
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
