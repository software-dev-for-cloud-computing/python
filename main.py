import asyncio
import logging
import os
import uuid
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from langchain.chains import llm

from app.business_logic.qa_process import QAProcess
from app.business_logic.upload_process import UploadProcess
from app.interfaces import retriever, prompts
from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.models.dto.documents import UploadDocumentRequestBody, UploadDocumentResponse
from app.models.dto.interfaces import InternalServerErrorResponse
from app.models.objects.chat_history_model import ChatHistory
from app.models.objects.llm_message_model import QAMessage
from app.services.llm_service import OpenAILLMModel
from app.services.pdf_reader_service import PDFReaderService
from app.services.qa_prompts_service import RagPromptsService
from app.services.retriever_service import QdrantRetriever
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


###
# Dependency injection for upload
###
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


###
# Dependency injection for QA
###
def get_llm(api_key: str) -> LlmModel:
    return OpenAILLMModel(api_key=api_key)


def get_retriever(api_key: str) -> Retriever:
    embedding_model = OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
        dimensions=VECTOR_STORE_DIMENSION
    )
    vector_store = QdrantVectorStore(embedding_model=embedding_model)
    return QdrantRetriever(vector_store)


def get_prompts() -> RagPrompts:
    return RagPromptsService()


@app.post("/document", response_model=UploadDocumentResponse | InternalServerErrorResponse)
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


@app.post("/qa", response_model=None)
async def qa_request(
        query: str,
        user_id: str,
        chat_history: List[QAMessage],
        conversation_id: str,
        document_id: Optional[str] = None,
        llm: LlmModel = Depends(get_llm),
        retriever: Retriever = Depends(get_retriever),
        prompts: RagPrompts = Depends(get_prompts),

):
    try:
        # messages = [message]
        chat_history = ChatHistory(messages=chat_history)

        response = await QAProcess.start_process(
            llm=llm,
            retriever=retriever,
            prompts=prompts,
            user_id=user_id,
            query=query,
            chat_history=chat_history,
            conversation_id=conversation_id
        )

        print("Response:")
        print(response)
        return response
        # return JSONResponse(content=response.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error | "
                                                    f"{e}")


async def main():
    query = "Wie gut ist Apple im KI Bereich?"
    user_id = "user_1"
    message = QAMessage(
        role="user",
        content="Hallo",
        user_id="user_1",
        conversation_id="1",
        timestamp="2024-08-02T22:09:51.825Z"
    )
    messages = [message]
    chat_history = ChatHistory(messages=messages)
    conversation_id = "1"

    llm = OpenAILLMModel(api_key="")
    retriever = QdrantRetriever(
        QdrantVectorStore(OpenAIEmbeddingModel(api_key="")))
    prompts = RagPromptsService()

    response = await QAProcess.start_process(
        llm=llm,
        retriever=retriever,
        prompts=prompts,
        user_id=user_id,
        query=query,
        chat_history=chat_history,
        conversation_id=conversation_id
    )

    print("Response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
