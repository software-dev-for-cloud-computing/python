import os
from typing import List, Optional

from fastapi.exceptions import RequestValidationError, HTTPException
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from app.core.domain.chunks.chunk_interface import ChunkInterface
from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.external_services.database.vector_store.vector_store_port import VectorStore
from app.models.dto.search import DocumentWithScore
from app.core.domain.chunks.chunk_model import ChunkModel, ChunkMetadata
from dotenv import load_dotenv
from app.core.utils.logger import Logger

load_dotenv()

logger = Logger('Logger')

COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION")
MAX_K_RESULTS = os.getenv("MAX_K_RESULTS")


class ChunkRepository(ChunkInterface):

    def __init__(self, vector_Store: VectorStore):
        self.vector_Store = vector_Store
        self.client = vector_Store.get_client()

    @logger.log_decorator(level="debug", message="Add chunks to collection")
    def add_chunks(self, chunks: List[ChunkModel], embedding_model: EmbeddingModel) -> bool:
        documents = [Document(page_content=chunk.content, metadata=chunk.metadata) for chunk in chunks]

        try:
            QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embedding_model.get_model(),
                url=os.getenv("VECTOR_STORE_URL"),
                collection_name=os.getenv("VECTOR_STORE_COLLECTION"),
                metadata_payload_key="metadata",
            )
            return True
        except:
            return False

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
        conditions = [models.FieldCondition(
            key="metadata.owner_id",
            match=models.MatchValue(value=user_id)
        )]

        if document_id is not None:
            conditions.append(models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=document_id)
            ))

        try:
            return self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=conditions,
                    )
                ),
            )
            # return SuccessResponse(message="Documents deleted successfully")
        except RequestValidationError:
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @logger.log_decorator(level="debug", message="Search for chunks")
    def search_chunks(self, embedding_model: EmbeddingModel, query: str, user_id: str,
                      document_id: Optional[str] = None,
                      k: int = MAX_K_RESULTS):
        try:
            connection = self.vector_Store.get_connection(embedding_model)

            if document_id is None:
                filter = {"owner_id": user_id}
            else:
                filter = {"owner_id": user_id, "document_id": document_id}

            results = connection.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
            )

            for result in results:
                document_data, score = result
                yield DocumentWithScore(
                    content=document_data.page_content,
                    metadata=ChunkMetadata(
                        document_id=document_data.metadata["document_id"],
                        owner_id=document_data.metadata["owner_id"],
                        page_number=document_data.metadata["page_number"],
                        on_page_index=document_data.metadata["on_page_index"],
                        conversation_id=document_data.metadata["conversation_id"]
                    ),
                    score=score
                )

        except RequestValidationError:
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))
