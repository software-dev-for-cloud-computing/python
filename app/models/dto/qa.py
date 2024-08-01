from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
from app.models.objects.llm_message_model import QAMessage


class RelatedDocument(BaseModel):
    documentId: str
    pageNumber: int
    score: float
    content: str


class QAMessageWithDocumentsResponse(BaseModel):
    role: str
    questionId: str
    conversationId: str
    answer: str
    timestamp: datetime
    relatedDocuments: List[RelatedDocument]
