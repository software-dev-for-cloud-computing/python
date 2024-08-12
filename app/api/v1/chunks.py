import asyncio
import logging
import os
import uuid
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.business_logic.upload_process import UploadProcess
from app.models.dto.documents import UploadDocumentRequestBody, UploadDocumentResponse
from app.models.dto.interfaces import InternalServerErrorResponse
from app.services.pdf_reader_service import PDFReaderService
from app.services.text_splitter_service import TextSplitterService
from app.services.embedding_service import OpenAIEmbeddingModel
from app.services.vector_store_service import VectorStoreQdrant
from app.interfaces.pdf_reader import PDFReader
from app.interfaces.text_splitter import TextSplitter
from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.vector_store import VectorStore
from app.utils.logger import Logger
from app.utils.logger import request_id_var

router = APIRouter()

load_dotenv()

logger = Logger(name="Logger")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_STORE_DIMENSION = os.getenv("EMBEDDING_DIMENSIONS")


def get_pdf_reader() -> PDFReader:
    return PDFReaderService()


def get_text_splitter() -> TextSplitter:
    return TextSplitterService(chunk_size=500)


def get_embedding_model(api_key: str = Header(..., alias="X-Api-Key")) -> EmbeddingModel:
    return OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
        dimensions=VECTOR_STORE_DIMENSION
    )


def get_vector_store() -> VectorStore:
    return VectorStoreQdrant()


@router.post("/document", response_model=UploadDocumentResponse | InternalServerErrorResponse)
async def upload_pdf(
        file: UploadFile = File(...),
        body: UploadDocumentRequestBody = Depends(),
        api_key: str = Header(..., alias="X-Api-Key"),
        pdf_reader: PDFReaderService = Depends(get_pdf_reader),
        text_splitter: TextSplitter = Depends(get_text_splitter),
        embedding_model: EmbeddingModel = Depends(get_embedding_model),
        vector_store: VectorStore = Depends(get_vector_store)
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    logger.log(level="debug", func_name="POST /document", message="Received request to upload document")

    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")

        response = await UploadProcess.start_process(
            request_id=request_id,
            body=body,
            file=file,
            pdf_reader=pdf_reader,
            text_splitter=text_splitter,
            embedding_model=embedding_model,
            vector_store=vector_store,
            document_id=body.document_id,
            owner_id=body.owner_id
        )

        # Log upload finished
        logger.log(level="debug", func_name="POST /document", message="Upload finished")

        return JSONResponse(content=response.dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error | {e}")


@router.delete("/api/v1/documents/{userId}/{documentId}")
async def delete_document(
        userId: str,
        documentId: str,
        vector_store: VectorStore = Depends(get_vector_store),
):
    try:
        response = vector_store.delete_chunks(document_id=documentId, user_id=userId)
        return JSONResponse(content=response.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/api/v1/documents/{userId}")
async def get_documents(
        userId: str,
        vector_store: VectorStore = Depends(get_vector_store),
):
    response = vector_store.get_all_chunks(user_id=userId)
    return response


@router.get("/api/v1/documents/{userId}/{documentId}")
async def get_document(
        userId: str,
        documentId: str,
        vector_store: VectorStore = Depends(get_vector_store),
):
    response = vector_store.get_chunks(document_id=documentId, user_id=userId)
    return response


@router.post("/api/v1/documents/search")
async def search_documents(
        query: str,
        userId: str,
        documentId: Optional[str] = None,
        k: int = 5,
        vector_store: VectorStore = Depends(get_vector_store),
        embedding_model: EmbeddingModel = Depends(get_embedding_model),
        api_key: str = Header(..., alias="X-Api-Key"),
):
    response = vector_store.search_chunks(
        embedding_model=embedding_model,
        query=query,
        user_id=userId,
        document_id=documentId,
        k=k
    )
    return response

    '''
        try:
        response = await vector_store.search_chunks(embedding_model=embedding_model,
                                                    query=query,
                                                    user_id=userId,
                                                    document_id=documentId,
                                                    k=k)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    '''



"""


@router.get("/documents")
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


@router.post("/documents/search_chunks")
async def search_chunks(
        query: Optional[str] = None,
        userId: Optional[str] = None,
        k: Optional[int] = 5):
    try:
        results: List[Tuple[Document, float]] = await search_process(question=query, user_id=userId,
                                                                     document_id=None, k=k)
        json_results = [search_results_to_dict(document=doc, score=score) for doc, score in results]
        return JSONResponse(content=json_results)
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
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


@router.delete("/documents")
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
"""
