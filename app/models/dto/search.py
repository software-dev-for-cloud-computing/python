from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
from app.models.objects.llm_message_model import QAMessage

'''
class RelatedDocument(BaseModel):
    documentId: str
    pageNumber: int
    score: float
    content: str


class SearchResponse(BaseModel):
    requestId: str
    relatedDocuments: List[RelatedDocument]
'''

