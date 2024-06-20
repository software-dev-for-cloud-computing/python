from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient as QdrantClientBase, models

from fastapi import Depends
from services.vectore_store.vector_store_service import VectorStoreService


class DocumentService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service

    def add_documents(self, documents: Document, embeddings):
        vector_store = self.vector_store_service.get_vectorstore()
        if not isinstance(documents, list) or not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("Input must be a list of Document objects.")
        vector_store.from_documents(documents,
                                    embeddings,
                                    url=self.vector_store_service.url,
                                    collection_name=self.vector_store_service.collection_name,
                                    metadata_payload_key="metadata",
                                    quantization_config=models.BinaryQuantization(
                                        binary=models.BinaryQuantizationConfig(always_ram=False, ), ),
                                    )

    def similarity_search(self, query, owner_id: str, document_id: str, k=5, return_metadata=True):
        vector_store = self.vector_store_service.get_vectorstore()
        if document_id is None:
            filter = {"owner_id": owner_id}
        else:
            filter = {"owner_id": owner_id, "document_id": document_id}
        return vector_store.similarity_search_with_score(query, k, filter=filter)

    def retrieve_documents(self, owner_id: str, document_id: str):
        vector_store = self.vector_store_service.get_vectorstore()

        if document_id:
            meta_data_key = "metadata.document_id"
            meta_data_value = document_id
        else:
            meta_data_key = "metadata.owner_id"
            meta_data_value = owner_id

        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # Number of documents to return; Default is 5
                "fetch_k": 20,  # Number of documents to pass into mmr algorithm; Default is 20
                "lambda_mult": 0.6,
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

    def get_documents(self, collection_name: str, document_id: Optional[str], user_id: Optional[str]):
        vector_store = self.vector_store_service.get_client()

        if document_id:
            meta_data_key = "metadata.document_id"
            meta_data_value = document_id
        else:
            meta_data_key = "metadata.owner_id"
            meta_data_value = user_id

        return vector_store.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key=meta_data_key, match=models.MatchValue(value=meta_data_value)),
                ]
            ),
            with_payload=True,
            with_vectors=False,
        )

    def delete_documents(self, collection_name: str, document_id: Optional[str], user_id: Optional[str]):
        vector_store = self.vector_store_service.get_client()

        if document_id:
            meta_data_key = "metadata.document_id"
            meta_data_value = document_id
        else:
            meta_data_key = "metadata.owner_id"
            meta_data_value = user_id

        return vector_store.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=meta_data_key,
                            match=models.MatchValue(value=meta_data_value),
                        ),
                    ],
                )
            ),
        )
