import os
import uuid
from typing import Optional
from dotenv import load_dotenv
from fastapi import File, UploadFile, HTTPException, Depends, Header, APIRouter
from fastapi.responses import JSONResponse

from app.core.external_services.embedding.openai_embedding_adapter import OpenAIEmbeddingModel
from app.core.services.upload_process import UploadProcess
from app.core.domain.chunks.chunk_repository import ChunkRepository
from app.core.domain.upload.pdf_reader_interface import PDFReader
from app.core.domain.upload.text_splitter_interface import TextSplitter
from app.exceptions.http_exceptions import HTTPInternalServerError
from app.models.dto.documents import UploadDocumentRequest, UploadDocumentResponse
from app.core.domain.upload.pdf_reader_service import PDFReaderService
from app.core.domain.upload.text_splitter_service import TextSplitterService
from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.external_services.database.vector_store.qdrant_vector_adapter import VectorStoreQdrant
from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.external_services.database.vector_store.vector_store_port import VectorStore
from app.core.utils.logger import Logger, request_id_var

router = APIRouter()

load_dotenv()

logger = Logger(name="Logger")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_STORE_DIMENSION = os.getenv("EMBEDDING_DIMENSIONS")


def get_pdf_reader() -> PDFReader:
    return PDFReaderService()


def get_text_splitter() -> TextSplitter:
    return TextSplitterService(chunk_size=500)


def get_embedding_model(api_key: str = Header(..., alias="x-api-key")) -> OpenAIEmbeddingModel:
    return OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
        dimensions=VECTOR_STORE_DIMENSION
    )


def get_vector_store() -> VectorStore:
    return VectorStoreQdrant()


def get_chunk_repository(vector_store: VectorStore = Depends(get_vector_store)) -> ChunkRepository:
    return ChunkRepository(vector_Store=vector_store)


@router.post("/document", response_model=UploadDocumentResponse)
async def upload_pdf(
        file: UploadFile = File(...),
        params: UploadDocumentRequest = Depends(),
        api_key: str = Header(..., alias="x-api-key"),
        pdf_reader: PDFReaderService = Depends(get_pdf_reader),
        text_splitter: TextSplitter = Depends(get_text_splitter),
        embedding_model: EmbeddingModel = Depends(get_embedding_model),
        chunk_repository: ChunkRepository = Depends(get_chunk_repository),
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    logger.log(level="debug", func_name="POST /document", message="Received request to upload document")

    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")

        response = await UploadProcess.start_process(
            request_id=request_id,
            params=params,
            file=file,
            pdf_reader=pdf_reader,
            text_splitter=text_splitter,
            embedding_model=embedding_model,
            chunk_repository=chunk_repository,
            document_id=params.document_id,
            owner_id=params.owner_id
        )

        logger.log(level="debug", func_name="POST /document", message="Upload finished")

        return JSONResponse(content=response.dict())

    except Exception as e:
        raise HTTPInternalServerError(error=str(e))


@router.delete("/api/v1/documents/{userId}/{documentId}")
async def delete_document(
        userId: str,
        documentId: str,
        chunk_repository: ChunkRepository = Depends(get_chunk_repository),
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    try:
        response = chunk_repository.delete_chunks(document_id=documentId, user_id=userId)
        return JSONResponse(content=response.dict())
    except Exception as e:
        raise HTTPInternalServerError(error=str(e))


@router.delete("/api/v1/documents/{userId}")
async def delete_document(
        userId: str,
        documentId: Optional[str] = None,
        chunk_repository: ChunkRepository = Depends(get_chunk_repository),
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    try:
        response = chunk_repository.delete_chunks(user_id=userId, document_id=documentId)
        return JSONResponse(content=response.dict())
    except Exception as e:
        raise HTTPInternalServerError(error=str(e))


@router.get("/api/v1/documents/{userId}")
async def get_documents(
        userId: str,
        chunk_repository: ChunkRepository = Depends(get_chunk_repository),
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    try:
        response = chunk_repository.get_all_chunks(user_id=userId)
        return response
    except Exception as e:
        raise HTTPInternalServerError(error=str(e))


@router.get("/api/v1/documents/{userId}/{documentId}")
async def get_document(
        userId: str,
        documentId: str,
        chunk_repository: ChunkRepository = Depends(get_chunk_repository),
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    try:
        response = chunk_repository.get_chunks(document_id=documentId, user_id=userId)
        return response
    except Exception as e:
        raise HTTPInternalServerError(error=str(e))


@router.post("/api/v1/documents/search")
async def search_documents(
        query: str,
        userId: str,
        documentId: Optional[str] = None,
        k: int = int(os.getenv("MAX_K_RESULTS")),
        chunk_repository: ChunkRepository = Depends(get_chunk_repository),
        embedding_model: EmbeddingModel = Depends(get_embedding_model),
        api_key: str = Header(..., alias="x-api-key"),
):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    try:
        response = chunk_repository.search_chunks(
            embedding_model=embedding_model,
            query=query,
            user_id=userId,
            document_id=documentId,
            k=k
        )

        return response
    except Exception as e:
        raise HTTPInternalServerError(error=str(e))