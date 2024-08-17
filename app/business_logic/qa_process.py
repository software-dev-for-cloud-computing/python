from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.models.objects.chat_history_model import ChatHistory
from app.models.objects.llm_message_model import LLMResponse
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
            api_key: str
    ) -> LLMResponse:
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
                                      user_id=user_id
                                      )

        return result
