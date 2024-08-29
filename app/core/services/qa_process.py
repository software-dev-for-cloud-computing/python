from app.core.domain.qa.qa_chain_service import QAChainService
from app.core.domain.qa.qa_models import QALLMResponse
from app.core.domain.qa.qa_prompts_interface import QAPromptsInterface
from app.core.external_services.llm.llm_port import LlmModel
from app.core.domain.retriever.retriever import Retriever
from app.models.objects.chat_history_model import ChatHistory
from app.core.utils.logger import Logger

logger = Logger('Logger')


class QAProcess:
    @staticmethod
    @logger.log_decorator(level="debug", message="Starting QA process")
    async def start_process(
            llm: LlmModel,
            retriever: Retriever,
            prompts: QAPromptsInterface,
            user_id: str,
            query: str,
            document_id: str,
            conversation_id: str,
            chat_history: ChatHistory
    ) -> QALLMResponse:
        chains = QAChainService()
        retriever_chain = chains.get_vector_store_retriever_chain(user_id=user_id,
                                                                  document_id=document_id,
                                                                  conversation_id=conversation_id,
                                                                  llm=llm,
                                                                  retriever=retriever,
                                                                  prompt=prompts)
        llm_chain = chains.get_llm_chain(llm=llm,
                                         prompt=prompts)
        rag_chain = chains.get_qa_chain(retriever_chain=retriever_chain,
                                        llm_chain=llm_chain)
        result = chains.run_qa_chain(rag_chain=rag_chain,
                                     query=query,
                                     chat_history=chat_history,
                                     user_id=user_id
                                     )

        return result
