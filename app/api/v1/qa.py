import os
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from app.business_logic.qa_process import QAProcess
from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.interfaces.vector_store import VectorStore
from app.models.dto.documents import UploadDocumentRequest
from app.models.dto.interfaces import InternalServerErrorResponse
from app.models.dto.qa import QAResponse, QARequest
from app.models.objects.chat_history_model import ChatHistory
from app.models.objects.llm_message_model import QAHistoryMessage
from app.services.llm_service import OpenAILLMModel
from app.services.qa_prompts_service import RagPromptsService
from app.services.retriever_service import QdrantRetriever
from app.services.embedding_service import OpenAIEmbeddingModel
from app.services.vector_store_service import VectorStoreQdrant
from app.exceptions.http_exceptions import HTTPInternalServerError
from app.utils.logger import Logger

router = APIRouter()

logger = Logger(name="Logger")


def get_embedding_model(api_key: str = Header(..., alias="x-api-key")) -> EmbeddingModel:
    return OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=os.getenv("EMBEDDING_MODEL"),
        dimensions=os.getenv("VECTOR_STORE_DIMENSION")
    )


def get_vector_store() -> VectorStore:
    return VectorStoreQdrant()


def get_llm(api_key: str = Header(..., alias="x-api-key")) -> LlmModel:
    return OpenAILLMModel(api_key=api_key)


def get_retriever(embedding_model: EmbeddingModel = Depends(get_embedding_model),
                  vector_store: VectorStore = Depends(get_vector_store)) -> Retriever:
    return QdrantRetriever(vector_store, embedding_model)


def get_prompts() -> RagPrompts:
    return RagPromptsService()


@router.post("/qa", response_model=QAResponse)
async def qa_request(
        chat_history: List[QAHistoryMessage],
        api_key: str = Header(..., alias="x-api-key"),
        query: str = Query(..., alias="query"),
        owner_id: str = Query(..., alias="ownerId"),
        conversation_id: str = Query(..., alias="conversationId"),
        document_id: Optional[str] = Query(None, alias="documentId"),
        llm: LlmModel = Depends(get_llm),
        embedding_model: EmbeddingModel = Depends(get_embedding_model),
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: Retriever = Depends(get_retriever),
        prompts: RagPrompts = Depends(get_prompts),
):
    logger.log(level="debug", func_name="POST /qa", message="QA request received")
    logger.log(level="debug", func_name="POST /qa", message=f"Query: {query}")
    logger.log(level="debug", func_name="POST /qa", message=f"Owner ID: {owner_id}")
    logger.log(level="debug", func_name="POST /qa", message=f"Conversation ID: {conversation_id}")
    logger.log(level="debug", func_name="POST /qa", message=f"Document ID: {document_id}")

    if document_id == "undefined":
        document_id = None
    # try:
    chat_history = ChatHistory(messages=chat_history)

    response = await QAProcess.start_process(
        llm=llm,
        retriever=retriever,
        prompts=prompts,
        user_id=owner_id,
        query=query,
        document_id=document_id,
        conversation_id=conversation_id,
        chat_history=chat_history,
        api_key=api_key,
    )

    return response
    '''
        except Exception as e:
        raise HTTPInternalServerError(
            error=str(e)
        )
    '''

