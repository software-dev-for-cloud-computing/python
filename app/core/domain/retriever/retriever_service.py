import os
from typing import Optional, List

from dotenv import load_dotenv

from app.core.external_services.database.vector_store.vector_store_port import VectorStore
from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.domain.retriever.retriever import Retriever
from qdrant_client import models

load_dotenv()

MAX_K_RESULTS = os.getenv("MAX_K_RESULTS")
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT"))


class QdrantRetriever(Retriever):

    def __init__(self,
                 vector_store: VectorStore,
                 embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def _filter_conditions(self,
                           user_id: str,
                           document_id: Optional[str] = None,
                           conversation_id: Optional[str] = None) -> List[models.FieldCondition]:

        must_conditions = []

        if user_id:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.owner_id",
                    match=models.MatchValue(value=user_id)
                )
            )

        if document_id:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=document_id)
                )
            )

        if conversation_id:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.conversation_id",
                    match=models.MatchValue(value=conversation_id)
                )
            )
        return must_conditions

    def get_mmr_retriever(self,
                          user_id: str,
                          document_id: Optional[str] = None,
                          conversation_id: Optional[str] = None,
                          k: int = int(MAX_K_RESULTS)):

        must_conditions = self._filter_conditions(user_id=user_id,
                                                  document_id=document_id,
                                                  conversation_id=conversation_id)

        vector_store_connection = self.vector_store.get_connection(embedding_model=self.embedding_model)
        return vector_store_connection.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,  # Number of documents to return; Default is 5
                "fetch_k": 20,  # Number of documents to pass into mmr algorithm; Default is 20
                "lambda_mult": LAMBDA_MULT,  # Diversity of Documents. Default = 0.5, Minimum = 1, Maximum = 0
                "filter": models.Filter(
                    must=must_conditions  # Filter for metadata
                )
            }
        )

    def get_similarity_retriever(self,
                                 user_id: str,
                                 document_id: Optional[str] = None,
                                 conversation_id: Optional[str] = None,
                                 k: int = int(MAX_K_RESULTS)):

        must_conditions = self._filter_conditions(user_id=user_id,
                                                  document_id=document_id,
                                                  conversation_id=conversation_id)

        vector_store_connection = self.vector_store.get_connection(embedding_model=self.embedding_model)
        return vector_store_connection.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,  # Number of documents to return; Default is 5
                "fetch_k": 20,  # Number of documents to pass into mmr algorithm; Default is 20
                "score_threshold": 0.8,
                "filter": models.Filter(
                    must=must_conditions  # Filter for metadata
                )
            }
        )
