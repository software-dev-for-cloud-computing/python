import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse

from app.business_logic.upload_process import UploadProcess
from app.models.dto.documents import UploadDocumentRequestBody
from app.services.pdf_reader_service import PDFReaderService
from app.services.text_splitter_service import TextSplitterService
from app.services.embedding_service import OpenAIEmbeddingModel
from app.services.vector_store_service import QdrantVectorStore
from app.interfaces.pdf_reader import PDFReader
from app.interfaces.text_splitter import TextSplitter
from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.vector_store import VectorStore
from app.utils.logger import Logger
from app.utils.logger import request_id_var

app = FastAPI()

load_dotenv()

logger = Logger(name="Logger")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_STORE_DIMENSION = os.getenv("EMBEDDING_DIMENSIONS")


# Dependency injection
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


def get_vector_store(embedding_model: EmbeddingModel = Depends(get_embedding_model)) -> VectorStore:
    return QdrantVectorStore(embedding_model=embedding_model)


@app.post("/document", response_model=None)
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
            vector_store=vector_store
        )

        # Log upload finished
        logger.log(level="debug", func_name="POST /document", message="Upload finished")

        return JSONResponse(content=response.dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
