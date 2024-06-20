from fastapi import Depends
from langchain_community.embeddings import OpenAIEmbeddings

from services.vectore_store.vector_store_service import VectorStoreService
from qdrant_client import models, QdrantClient


class CollectionService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service

    def create_collection(self, collection_name: str):
        vector_store = self.vector_store_service.get_client()
        vector_store.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE, on_disk=True),
            hnsw_config=models.HnswConfigDiff(m=64, ef_construct=512, on_disk=True),
        )
        if vector_store.collection_exists(collection_name=collection_name):
            return {"message": f"Collection {collection_name} created successfully"}
        else:
            return {"message": f"Creating Collection {collection_name} failed"}

    def delete_collection(self, collection_name: str):
        vector_store = self.vector_store_service.get_client()
        vector_store.delete_collection(collection_name=collection_name)

        if vector_store.collection_exists(collection_name=collection_name):
            return {"message": f"Deleting Collection {collection_name} failed"}
        else:
            return {"message": f"Collection {collection_name} deleted successfully"}

    def get_collections(self):
        vector_store = self.vector_store_service.get_client()
        return vector_store.get_collections()

    def get_collection(self, collection_name: str):
        vector_store = self.vector_store_service.get_client()
        return vector_store.get_collection(collection_name=collection_name)


'''
    def get_collections(self):
        client = QdrantClient(url="http://localhost:6333")

        client.get_collections()

        client.delete_collection(collection_name="CoStudy")
        #client.get_collection(collection_name="CoStudy")

        return client.get_collection(collection_name="CoStudy")
'''
