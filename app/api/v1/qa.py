import os
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Header
from app.business_logic.qa_process import QAProcess
from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.interfaces.vector_store import VectorStore
from app.models.objects.chat_history_model import ChatHistory
from app.models.objects.llm_message_model import QAMessage
from app.services.llm_service import OpenAILLMModel
from app.services.qa_prompts_service import RagPromptsService
from app.services.retriever_service import QdrantRetriever
from app.services.embedding_service import OpenAIEmbeddingModel
from app.services.vector_store_service import VectorStoreQdrant
from app.utils.logger import Logger

router = APIRouter()


def get_embedding_model(api_key: str = Header(..., alias="X-Api-Key")) -> EmbeddingModel:
    return OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=os.getenv("EMBEDDING_MODEL"),
        dimensions=os.getenv("VECTOR_STORE_DIMENSION")
    )


def get_vector_store() -> VectorStore:
    return VectorStoreQdrant()


def get_llm(api_key: str = Header(..., alias="X-Api-Key")) -> LlmModel:
    return OpenAILLMModel(api_key=api_key)


def get_retriever(embedding_model: EmbeddingModel = Depends(get_embedding_model),
                  vector_store: VectorStore = Depends(get_vector_store)) -> Retriever:
    return QdrantRetriever(vector_store, embedding_model)


def get_prompts() -> RagPrompts:
    return RagPromptsService()


@router.post("/qa", response_model=None)
async def qa_request(
        query: str,
        user_id: str,
        chat_history: List[QAMessage],
        api_key: str = Header(..., alias="X-Api-Key"),
        document_id: Optional[str] = None,
        llm: LlmModel = Depends(get_llm),
        embedding_model: EmbeddingModel = Depends(get_embedding_model),
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: Retriever = Depends(get_retriever),
        prompts: RagPrompts = Depends(get_prompts),
):
    try:
        chat_history = ChatHistory(messages=chat_history)

        response = await QAProcess.start_process(
            llm=llm,
            retriever=retriever,
            prompts=prompts,
            user_id=user_id,
            query=query,
            chat_history=chat_history,
            api_key=api_key,
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error | {e}")
