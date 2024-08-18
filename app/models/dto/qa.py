from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.dto.interfaces import SuccessResponse
from app.models.objects.chunk_model import ChunkModel

from typing import Optional
from fastapi import Query
from pydantic import BaseModel


class QARequest(BaseModel):
    query: str = Query(..., alias="query")
    owner_id: str = Query(..., alias="ownerId")
    document_id: Optional[str] = Query(None, alias="documentId")
    conversation_id: str = Query(..., alias="conversationId")


class QAResponse(SuccessResponse):
    question: str
    answer: str
    related_documents: List[ChunkModel] = Field(..., alias="related_documents")
