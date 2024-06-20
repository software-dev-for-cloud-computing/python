from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory


class MemoryChainManager:
    def __init__(self, llm, prompt_template):
        self.llm = llm
        # self.prompt_template = prompt_template
        self.memory_chain = None
        self.store = {}

    def get_system_prompt_template(self):
        system_prompt = """You are a helpful assistant for question-answering tasks. 
                           Use only the following pieces of retrieved context to answer the question. 
                           Don’t justify your answers. 
                           Don’t give information not mentioned in the CONTEXT INFORMATION.
                           If you don't know the answer, just say that you don't know. 
                           Use five sentences maximum and keep the answer concise."""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        return prompt
    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = ChatMessageHistory()
        return self.store[(user_id, conversation_id)]

    def get_runnable(self, prompt, llm):
        return prompt | llm

    def ask_question(self, runnable):
        return RunnableWithMessageHistory(
            runnable,
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

'''
    def get_memory_chain(self):
        if self.memory_chain is None:
            #self.set_prompt_template()
            memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
            self.memory_chain = ConversationChain(
                llm=self.llm,
                prompt=self.prompt_template,
                memory=memory,
                verbose=True
            )
        return self.memory_chain
'''

