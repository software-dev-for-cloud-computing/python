from pydantic import BaseModel, PositiveInt, Field, field_validator
from app.exceptions.exceptions import InvalidOwnerIdError, InvalidDocumentIdError, InvalidConversationIdError
from app.core.domain.chunks.chunk_exceptions import InvalidContentError, InvalidPageNumberError


class ChunkMetadata(BaseModel):
    document_id: str = Field(strict=True)
    conversation_id: str = Field(strict=True)
    owner_id: str = Field(strict=True)
    page_number: PositiveInt = Field(strict=True)
    on_page_index: PositiveInt = Field(strict=True)

    @field_validator('document_id', mode='before')
    def validate_document_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise InvalidDocumentIdError()
        return value

    @field_validator('conversation_id', mode='before')
    def validate_conversation_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise InvalidConversationIdError()
        return value

    @field_validator('owner_id', mode='before')
    def validate_owner_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise InvalidOwnerIdError()
        return value

    @field_validator('page_number', mode='before')
    def validate_page_number(cls, value: PositiveInt) -> PositiveInt:
        if value <= 0:
            raise InvalidPageNumberError()
        return value


class ChunkModel(BaseModel):
    content: str = Field(min_length=10, max_length=10000, strict=True, description="The content of the chunk")
    metadata: ChunkMetadata = Field(strict=True, description="The metadata of the chunk")

    @field_validator('content', mode='before')
    def clean_and_validate_content(cls, value: str) -> str:

        # Validate content
        if not isinstance(value, str):
            raise InvalidContentError()
        if len(value) < 10:
            raise InvalidContentError("Content must be at least 10 characters long")

        # Remove unwanted characters
        value = value.replace("\n\n", " ").replace("\n", " ").replace("\"", "").replace("  ", " ")

        return value
