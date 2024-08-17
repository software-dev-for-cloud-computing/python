from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.interfaces.chains import RagChains
from app.interfaces.llm_model import LlmModel
from app.interfaces.prompts import RagPrompts
from app.interfaces.retriever import Retriever
from app.models.objects.chat_history_model import ChatHistory
from app.models.objects.chunk_model import ChunkModel, ChunkMetadata
from app.models.objects.llm_message_model import LLMResponse


class RagChainService(RagChains):
    def __init__(self):
        pass

    def get_vector_store_retriever_chain(self, user_id: str, llm: LlmModel, retriever: Retriever, prompt: RagPrompts):
        return create_history_aware_retriever(
            llm=llm.get_llm(),
            retriever=retriever.get_document_retriever(user_id=user_id, document_id=None, k=5),
            prompt=prompt.get_history_aware_prompt_for_vector_store_retriever()
        )

    def get_llm_chain(self, llm, prompt):
        return create_stuff_documents_chain(llm=llm.get_llm(),
                                            prompt=prompt.get_llm_prompt_with_context())

    def get_rag_chain(self, retriever_chain, llm_chain):
        return create_retrieval_chain(
            retriever=retriever_chain,
            combine_docs_chain=llm_chain
        )

    def run_rag_chain(self, rag_chain, query, chat_history: ChatHistory, user_id) -> LLMResponse:
        chat_history = chat_history.get_langchain_base_chat_message_history().messages

        response = rag_chain.invoke(
            {"input": query, "chat_history": chat_history}
        )

        llm_response: LLMResponse = LLMResponse(
            question=query,
            answer=response["answer"],
            related_documents=[
                ChunkModel(
                    content=doc.page_content,
                    metadata=ChunkMetadata(
                        document_id=doc.metadata["document_id"],
                        owner_id=doc.metadata["owner_id"],
                        page_number=doc.metadata["page_number"],
                        on_page_index=doc.metadata["on_page_index"]
                    )
                ) for doc in response["context"]
            ]
        )

        return llm_response
