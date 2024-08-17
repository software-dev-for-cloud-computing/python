from typing import List

from app.models.dto.interfaces import SuccessResponse
from app.models.objects.chunk_model import ChunkModel


class QAResponse(SuccessResponse):
    question: str
    answer: str
    related_documents: List[ChunkModel]
