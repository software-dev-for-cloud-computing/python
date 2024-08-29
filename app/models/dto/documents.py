from typing import List

from pydantic import Field, BaseModel

from app.models.dto.interfaces import SuccessResponse
from app.core.domain.chunks.chunk_model import ChunkModel


class UploadDocumentRequest(BaseModel):
    owner_id: str = Field(..., alias="ownerId")
    document_id: str = Field(..., alias="documentId")
    conversation_id: str = Field(..., alias="conversationId")


class UploadDocumentResponse(SuccessResponse):
    document_id: str = Field(..., alias="documentId")
    owner_id: str = Field(..., alias="ownerId")
    total_number_of_chunks_created: int = Field(..., alias="totalNumberOfChunksCreated")
    total_number_of_tokens_used: int = Field(..., alias="totalNumberOfTokensUsed")


class DeleteDocumentResponse(SuccessResponse):
    document_id: str = Field(..., alias="documentId")


class GetDocumentResponse(SuccessResponse):
    document_id: str = Field(..., alias="documentId")
    document: ChunkModel


class GetDocumentsByIdResponse(SuccessResponse):
    document_id: str = Field(..., alias="documentId")
    document: List[ChunkModel]
