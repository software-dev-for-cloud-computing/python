from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.domain.qa.qa_prompts_interface import QAPromptsInterface

VECTOR_STORE_RETRIEVER_PROMPT = """Given a chat history and the latest user question which might 
        reference context in the chat history, formulate a standalone question which can be understood without the chat 
        history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

LLM_PROMPT_WITH_CONTEXT = """You are a helpful assistant for question-answering tasks.
                   Use only the following pieces of retrieved context to answer the question.
                   Don’t justify your answers.
                   Don’t give information not mentioned in the CONTEXT INFORMATION.
                   If you don't know the answer, just say that you don't know.
                   Use five sentences maximum and keep the answer concise.
                   {context}"""


class QAPromptsService(QAPromptsInterface):
    def __init__(self):
        self._history_aware_prompt = None
        self._llm_prompt_with_context = None

    def get_history_aware_prompt_for_vector_store_retriever(self) -> ChatPromptTemplate:
        """Returns the history-aware prompt for the vector store retriever."""
        if not self._history_aware_prompt:
            self.set_history_aware_prompt_for_vector_store_retriever()
        return self._history_aware_prompt

    def set_history_aware_prompt_for_vector_store_retriever(self, prompt: str = VECTOR_STORE_RETRIEVER_PROMPT):
        """Sets the history-aware prompt for the vector store retriever."""
        self._history_aware_prompt = ChatPromptTemplate.from_messages(
            [("system", prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

    def get_llm_prompt_with_context(self) -> ChatPromptTemplate:
        """Returns the LLM prompt with context."""
        if not self._llm_prompt_with_context:
            self.set_llm_prompt_with_context()
        return self._llm_prompt_with_context

    def set_llm_prompt_with_context(self, prompt: str = LLM_PROMPT_WITH_CONTEXT):
        """Sets the LLM prompt with context."""
        self._llm_prompt_with_context = ChatPromptTemplate.from_messages(
            [("system", prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
