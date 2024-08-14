from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

from app.models.objects.chunk_model import ChunkModel
from app.models.objects.llm_message_model import QAMessage


class DocumentWithScore(ChunkModel):
    score: float


class SearchResponse(BaseModel):
    requestId: str
    relatedDocuments: List[DocumentWithScore]
