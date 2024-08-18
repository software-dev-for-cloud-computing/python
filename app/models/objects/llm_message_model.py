import os
from typing import Literal, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from app.models.objects.chunk_model import ChunkModel

load_dotenv()


class QAHistoryMessage(BaseModel):
    role: Literal['user', 'ai']
    content: str = Field(...,
                         min_length=os.getenv("MIN_LENGTH_CONTEXT_MESSAGE"),
                         max_length=os.getenv("MAX_LENGTH_CONTEXT_MESSAGE"))
    user_id: str
    timestamp: str


class LLMResponse(BaseModel):
    question: str
    answer: str
    related_documents: List[ChunkModel]
