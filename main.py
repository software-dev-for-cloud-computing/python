from typing import List, Tuple, Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Body, Depends, Form
from fastapi.exceptions import RequestValidationError, HTTPException
from langchain_core.documents import Document
from starlette.responses import JSONResponse

from core.config import QDRANT_HOST, QDRANT_COLLECTION_NAME
from services.manager import upload_process, search_process, retrieve_process
from services.vectore_store.document_service import DocumentService
from services.vectore_store.vector_store_service import VectorStoreService
from utils.serializer_helper import search_results_to_dict
from services.vectore_store.collection_service import CollectionService

app = FastAPI()


@app.get("/collections")
async def get_collections():
    try:
        vector_store_service = VectorStoreService()
        collection_service = CollectionService(vector_store_service)
        collections = collection_service.get_collections()
        return {"collections": collections}
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collectionName}")
async def get_collection(collectionName: str):
    try:
        vector_store_service = VectorStoreService()
        collection_service = CollectionService(vector_store_service)
        collection = collection_service.get_collection(collection_name=collectionName)
        return {"collection": collection}
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
async def create_collection(collectionName: str = Form(...)):
    try:
        vector_store_service = VectorStoreService()
        collection_service = CollectionService(vector_store_service)
        response = collection_service.create_collection(collectionName)
        return JSONResponse(content=response)
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections")
async def delete_collection(collectionName: str = Form(...)):
    try:
        vector_store_service = VectorStoreService()
        collection_service = CollectionService(vector_store_service)
        response = collection_service.delete_collection(collection_name=collectionName)
        return JSONResponse(content=response)
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), userId: str = Form(...),
                 documentId: str = Form(...)):
    try:
        data = await file.read()
        background_tasks.add_task(upload_process, data=data, document_id=documentId, user_id=userId)
        return {"message": "File uploaded successfully"}
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")


@app.post("/documents/search")
async def search(query: Optional[str] = None, userId: Optional[str] = None, k: Optional[int] = 5):
    try:
        results: List[Tuple[Document, float]] = await search_process(question=query, user_id=userId,
                                                                     document_id=None, k=k)
        json_results = [search_results_to_dict(document=doc, score=score) for doc, score in results]
        return JSONResponse(content=json_results)
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def get_documents(userId: Optional[str] = None, documentId: Optional[str] = None):
    try:
        vector_store_client = VectorStoreService()
        vector_store_client.connect(collection_name=QDRANT_COLLECTION_NAME)
        document_service = DocumentService(vector_store_service=vector_store_client)
        if documentId:
            documents = document_service.get_documents(collection_name=QDRANT_COLLECTION_NAME,
                                                       user_id=None,
                                                       document_id=documentId)
        else:
            documents = document_service.get_documents(collection_name=QDRANT_COLLECTION_NAME,
                                                       user_id=userId,
                                                       document_id=None)
        return {"documents": documents}
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def delete_documents(userId: Optional[str] = None, documentId: Optional[str] = None):
    try:
        vector_store_client = VectorStoreService()
        vector_store_client.connect(collection_name=QDRANT_COLLECTION_NAME)
        document_service = DocumentService(vector_store_service=vector_store_client)
        if documentId:
            documents = document_service.delete_documents(collection_name=QDRANT_COLLECTION_NAME,
                                                          user_id=None,
                                                          document_id=documentId)
        else:
            documents = document_service.delete_documents(collection_name=QDRANT_COLLECTION_NAME,
                                                          user_id=userId,
                                                          document_id=None)
        return {"documents": documents}
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(query: str, userId: str, conversationId: str, documentId: Optional[str] = None):
    try:
        results = await retrieve_process(query=query, user_id=userId, conversation_id=conversationId, document_id=documentId)
        return results
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
