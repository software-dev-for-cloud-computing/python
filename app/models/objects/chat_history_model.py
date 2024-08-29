from pydantic import BaseModel
from typing import List
from langchain_community.chat_message_histories import ChatMessageHistory
from app.core.domain.qa.qa_models import QAHistoryMessage


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

