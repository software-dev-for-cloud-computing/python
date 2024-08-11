import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime

load_dotenv()

class QAMessage(BaseModel):
    role: Literal['user', 'ai']
    content: str = Field(...,
                         min_length=os.getenv("MIN_LENGTH_CONTEXT_MESSAGE"),
                         max_length=os.getenv("MAX_LENGTH_CONTEXT_MESSAGE"))
    user_id: str
    conversation_id: str
    timestamp: str
