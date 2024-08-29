from abc import ABC, abstractmethod

from langchain_core.prompts import ChatPromptTemplate


class QAPromptsInterface(ABC):

    @abstractmethod
    def get_history_aware_prompt_for_vector_store_retriever(self) -> ChatPromptTemplate:
        pass

    @abstractmethod
    def set_history_aware_prompt_for_vector_store_retriever(self, prompt: str):
        pass

    @abstractmethod
    def get_llm_prompt_with_context(self) -> ChatPromptTemplate:
        pass

    @abstractmethod
    def set_llm_prompt_with_context(self, prompt: str):
        pass

