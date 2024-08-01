from pydantic import BaseModel
from typing import List

from app.models.objects.llm_message_model import QAMessage


class MessageHistoryModel(BaseModel):
    messages: List[QAMessage]