import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from app.exceptions.http_exceptions import HTTPInternalServerError
from app.core.external_services.database.vector_store.vector_store_port import VectorStore
from app.core.external_services.database.vector_store.qdrant_vector_adapter import VectorStoreQdrant
from app.core.utils.logger import Logger

router = APIRouter()

logger = Logger(name="Logger")

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_STORE_DIMENSION = os.getenv("EMBEDDING_DIMENSIONS")


def get_vector_store() -> VectorStore:
    return VectorStoreQdrant()

@router.get("/collections")
async def get_collections(vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        collections = vector_store.get_collections()
        return collections
    except Exception as e:
        raise HTTPInternalServerError(
            error=str(e)
        )


@router.get("/collections/{collection_name}")
async def get_collection(collection_name: str, vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        collection_info = vector_store.get_collection(collection_name)
        return collection_info
    except Exception as e:
        raise HTTPInternalServerError(
            error=str(e)
        )


@router.post("/collections/{collection_name}")
async def create_collection(collection_name: str, vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        vector_store.create_collection(collection_name)
        return {"message": f"Collection '{collection_name}' created successfully."}
    except Exception as e:
        raise HTTPInternalServerError(
            error=str(e)
        )


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        vector_store.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully."}
    except Exception as e:
        raise HTTPInternalServerError(
            error=str(e)
        )
