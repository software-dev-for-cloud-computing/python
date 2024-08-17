from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import Runnable

from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.models.objects.llm_message_model import LLMResponse


class RagChains(ABC):
    @abstractmethod
    def get_vector_store_retriever_chain(self, user_id: str, llm: LlmModel, retriever: Retriever,
                                         prompt: RagPrompts) -> Runnable:
        pass

    @abstractmethod
    def get_llm_chain(self, llm: LlmModel, prompt: RagPrompts) -> Runnable:
        pass

    @abstractmethod
    def get_rag_chain(self, retriever_chain: Runnable, llm_chain: Runnable) -> Runnable:
        pass

    @abstractmethod
    def run_rag_chain(self,
                      rag_chain: Runnable,
                      query: str,
                      chat_history: Any,
                      user_id: str,
                      ) -> LLMResponse:
        pass
