import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Header

from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.vector_store import VectorStore
from app.services.embedding_service import OpenAIEmbeddingModel
from app.services.vector_store_service import VectorStoreQdrant
from app.utils.logger import Logger

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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}")
async def get_collection(collection_name: str, vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        collection_info = vector_store.get_collection(collection_name)
        return collection_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{collection_name}")
async def create_collection(collection_name: str, vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        vector_store.create_collection(collection_name)
        return {"message": f"Collection '{collection_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, vector_store: VectorStoreQdrant = Depends(get_vector_store)):
    try:
        vector_store.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
