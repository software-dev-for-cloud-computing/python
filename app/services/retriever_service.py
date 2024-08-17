import os
from typing import Optional

from dotenv import load_dotenv

from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.retriever import Retriever
from qdrant_client import models

from app.interfaces.vector_store import VectorStore

load_dotenv()

MAX_K_RESULTS = 5 #os.getenv("MAX_K_RESULTS")
lambda_mult = 0.4


class QdrantRetriever(Retriever):

    def __init__(self,
                 vector_store: VectorStore,
                 embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def get_document_retriever(self,
                               user_id: str,
                               document_id: Optional[str] = None,
                               conversation_id: Optional[str] = None,
                               k: int = int(MAX_K_RESULTS)):

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

        vector_store_connection = self.vector_store.get_connection(embedding_model=self.embedding_model)
        return vector_store_connection.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,  # Number of documents to return; Default is 5
                "fetch_k": 20,  # Number of documents to pass into mmr algorithm; Default is 20
                "lambda_mult": lambda_mult,  # Diversity of Documents. Default = 0.5, Minimum = 1, Maximum = 0
                "filter": models.Filter(
                    must=must_conditions  # Filter for metadata
                )
            }
        )
