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

    def run_rag_chain(self, rag_chain, query, chat_history: ChatHistory, user_id, conversation_id):
        print("Running RAG chain")
        # Create the callable for getting chat history
        history_callable = chat_history.get_langchain_base_chat_message_history_callable()

        runner = RunnableWithMessageHistory(
            rag_chain,
            history_callable,
            input_messages_key="input",
            history_messages_key="chat_history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

        llm_response = runner.invoke(
            {"input": query, "chat_history": chat_history},
            config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
        )

        return llm_response
