from pydantic import BaseModel
from typing import List
from app.models.objects.llm_message_model import QAMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class ChatHistory(BaseModel):
    messages: List[QAMessage]

    def get_langchain_base_chat_message_history(self, conversation_id: str, user_id: str):
        history = ChatMessageHistory()
        for msg in self.messages:
            if msg.conversation_id == conversation_id and msg.user_id == user_id:
                if msg.role == 'user':
                    history.add_user_message(msg.content)
                elif msg.role == 'ai':
                    history.add_ai_message(msg.content)
        return history

    def get_langchain_base_chat_message_history_callable(self):
        return lambda conversation_id, user_id: self.get_langchain_base_chat_message_history(conversation_id, user_id)



'''
class InMemoryHistory:
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



class MessageHistoryModel(BaseModel):
    messages: List[QAMessage]

    def get_langchain_base_chat_message_history(self, conversation_id: str, user_id: str):
        chat_history = ChatMessageHistory()
        print("Starting with chat history")
        for message in self.messages:
            chat_message = BaseMessage(
                role=message.role,
                content=message.content,
                user_id=user_id,
                conversation_id=conversation_id,
                timestamp=message.timestamp
            )
            chat_history.add_message(chat_message)
        print(chat_history)
        return chat_history

    def get_langchain_base_chat_message_history_callable(self, conversation_id: str, user_id: str):
        return lambda: self.get_langchain_base_chat_message_history(conversation_id, user_id)

'''
