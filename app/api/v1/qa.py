import os
from typing import Optional, List

from fastapi import APIRouter, Depends, Header, Query

from app.core.external_services.database.vector_store.qdrant_vector_adapter import VectorStoreQdrant
from app.core.external_services.embedding.openai_embedding_adapter import OpenAIEmbeddingModel
from app.core.external_services.llm.openai_llm_adapter import OpenAILLMModel
from app.core.services.qa_process import QAProcess
from app.core.domain.qa.qa_models import QAHistoryMessage
from app.core.domain.qa.qa_prompts_interface import QAPromptsInterface
from app.core.domain.qa.qa_prompts_service import QAPromptsService
from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.core.external_services.llm.llm_port import LlmModel
from app.core.domain.retriever.retriever import Retriever
from app.core.external_services.database.vector_store.vector_store_port import VectorStore
from app.models.dto.qa import QAResponse
from app.models.objects.chat_history_model import ChatHistory
from app.core.domain.retriever.retriever_service import QdrantRetriever
from app.core.utils.logger import Logger

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


def get_prompts() -> QAPromptsInterface:
    return QAPromptsService()


@router.post("/qa", response_model=QAResponse)
async def qa_request(
        chat_history: List[QAHistoryMessage],
        api_key: str = Header(..., alias="x-api-key"),
        query: str = Query(..., alias="query"),
        owner_id: str = Query(..., alias="ownerId"),
        conversation_id: str = Query(..., alias="conversationId"),
        document_id: Optional[str] = Query(None, alias="documentId"),
        llm: LlmModel = Depends(get_llm),
        retriever: Retriever = Depends(get_retriever),
        prompts: QAPromptsInterface = Depends(get_prompts),
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
        chat_history=chat_history
        # api_key=api_key,
    )

    return response
    '''
        except Exception as e:
        raise HTTPInternalServerError(
            error=str(e)
        )
    '''

