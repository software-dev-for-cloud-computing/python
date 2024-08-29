from pydantic import BaseModel, Field
from typing import List

from app.core.domain.chunks.chunk_model import ChunkModel


class DocumentWithScore(ChunkModel):
    score: float


class SearchResponse(BaseModel):
    requestId: str
    related_documents: List[DocumentWithScore] = Field(..., alias="relatedDocuments")
