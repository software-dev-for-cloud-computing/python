import os
from typing import Optional

from dotenv import load_dotenv

from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.retriever import Retriever
from qdrant_client import models

from app.interfaces.vector_store import VectorStore

load_dotenv()

MAX_K_RESULTS = os.getenv("MAX_K_RESULTS")


class QdrantRetriever(Retriever):

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def get_document_retriever(self, user_id: str, document_id: Optional[str] = None, k: int = MAX_K_RESULTS):
        if document_id:
            meta_data_key = "metadata.document_id"
            meta_data_value = document_id
        else:
            meta_data_key = "metadata.owner_id"
            meta_data_value = user_id

        vector_store_connection = self.vector_store.get_connection(embedding_model=self.embedding_model)
        return vector_store_connection.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,  # Number of documents to return; Default is 5
                "fetch_k": 20,  # Number of documents to pass into mmr algorithm; Default is 20
                "lambda_mult": 0.4, # Wie stark die Diversität der Ergebnisse berücksichtigt wird. 0.5 ist der Standardwert. 1 Minimum, 0 Maximum.
                "filter": models.Filter(
                    must=[
                        models.FieldCondition(
                            key=meta_data_key,
                            match=models.MatchValue(value=meta_data_value)
                        )
                    ]
                )
                # Diversity of results returned by MMR; 1 for min diversity and 0 for max; Default: 0.5
                # "filter": filter_criteria  # Filter for metadata
            }
        )

    def search(self, query: str, user_id: str, document_id: str, k: int):
        pass
