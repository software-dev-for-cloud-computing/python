from datetime import datetime
from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import Field, BaseModel

from utils.singleton import SingletonMeta


class InMemoryHistory(metaclass=SingletonMeta):
    """In memory implementation of chat message history."""

    def __init__(self):
        self.messages = []

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> List[BaseMessage]:
        """Return messages as a list"""
        return self.messages


class ChainManager:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.store = {}

    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryHistory()
        return self.store[(user_id, conversation_id)]

    def get_rag_chain(self):
        # -------------------------------------------------------------------------------------------------------------
        # DEFINE THE CONTEXT CHAIN
        # -------------------------------------------------------------------------------------------------------------
        context_prompt_text = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""

        context_prompt = ChatPromptTemplate.from_messages(
            [("system", context_prompt_text), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        context_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=context_prompt)
        # -------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------------------
        # DEFINE THE QA CHAIN
        # -------------------------------------------------------------------------------------------------------------
        qa_prompt_text = """You are a helpful assistant for question-answering tasks. 
                           Use only the following pieces of retrieved context to answer the question. 
                           Don’t justify your answers. 
                           Don’t give information not mentioned in the CONTEXT INFORMATION.
                           If you don't know the answer, just say that you don't know. 
                           Use five sentences maximum and keep the answer concise.
                           {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", qa_prompt_text), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        qa_chain = create_stuff_documents_chain(llm=self.llm,
                                                prompt=qa_prompt)

        rag_chain = create_retrieval_chain(
            retriever=context_chain,
            combine_docs_chain=qa_chain
        )

        return rag_chain

    def run(self, question, user_id: str, conversation_id: str):
        history_instance = self.store.get((user_id, conversation_id), InMemoryHistory())
        history_instance.add_message({"role": "user", "content": question, "user_id": user_id, "conversation_id": conversation_id,"timestamp": datetime.now()})
        chat_history = self.get_session_history(user_id=user_id, conversation_id=conversation_id).get_messages()
        last_5_messages = chat_history[-5:]

        runner = RunnableWithMessageHistory(
            self.get_rag_chain(),
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
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

        # Convert the chat history to the expected list format
        #chat_history = self.get_session_history(user_id=user_id, conversation_id=conversation_id).get_messages()


        print(last_5_messages)

        response = runner.invoke(
            {"input": question, "chat_history": chat_history},
            config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
        )

        history_instance.add_message({"role": "ai", "content": response["answer"], "user_id": user_id, "conversation_id": conversation_id,"timestamp": datetime.now()})

        return response
