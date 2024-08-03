from fastapi.exceptions import RequestValidationError, HTTPException

from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.interfaces.vector_store import VectorStore
from app.models.objects.chat_history_model import ChatHistory
from app.services.chains_service import RagChainService


class QAProcess:
    @staticmethod
    async def start_process(
        llm: LlmModel,
        retriever: Retriever,
        prompts: RagPrompts,
        user_id: str,
        query: str,
        chat_history: ChatHistory,
        conversation_id: str,
    ):
        chains = RagChainService()
        retriever_chain = chains.get_vector_store_retriever_chain(user_id=user_id,
                                                                  llm=llm,
                                                                  retriever=retriever,
                                                                  prompt=prompts)
        llm_chain = chains.get_llm_chain(llm=llm,
                                         prompt=prompts)
        rag_chain = chains.get_rag_chain(retriever_chain=retriever_chain,
                                         llm_chain=llm_chain)
        result = chains.run_rag_chain(rag_chain=rag_chain,
                                      query=query,
                                      chat_history=chat_history,
                                      user_id=user_id,
                                      conversation_id=conversation_id)
        return result


