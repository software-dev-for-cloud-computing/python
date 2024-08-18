from pydantic import BaseModel
from typing import List
from app.models.objects.llm_message_model import QAHistoryMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class ChatHistory(BaseModel):
    messages: List[QAHistoryMessage]

    def get_langchain_base_chat_message_history(self):
        history = ChatMessageHistory()
        for msg in self.messages:
            if msg.role == 'user':
                history.add_user_message(msg.content)
            elif msg.role == 'ai':
                history.add_ai_message(msg.content)
        return history

