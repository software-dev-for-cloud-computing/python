import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.runnables import Runnable
from pydantic import PositiveInt

from app.core.external_services.llm.llm_port import LlmModel
from app.core.domain.qa.qa_models import QALLMResponse
from app.core.domain.qa.qa_prompts_interface import QAPromptsInterface
from app.core.domain.retriever.retriever import Retriever


class QAChains(ABC):
    @abstractmethod
    def get_vector_store_retriever_chain(self, user_id: str,
                                         llm: LlmModel,
                                         retriever: Retriever,
                                         prompt: QAPromptsInterface,
                                         document_id: Optional[str] = None,
                                         conversation_id: Optional[str] = None,
                                         k: PositiveInt = int(os.getenv("MAX_K_RESULTS"))
                                         ) -> Runnable:
        pass

    @abstractmethod
    def get_llm_chain(self, llm: LlmModel, prompt: QAPromptsInterface) -> Runnable:
        pass

    @abstractmethod
    def get_qa_chain(self, retriever_chain: Runnable, llm_chain: Runnable) -> Runnable:
        pass

    @abstractmethod
    def run_qa_chain(self,
                      qa_chain: Runnable,
                      query: str,
                      chat_history: Any,
                      user_id: str,
                      ) -> QALLMResponse:
        pass
