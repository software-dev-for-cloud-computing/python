import uuid
from pydantic import BaseModel, Field
from fastapi import status


class ApiResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="requestId")
    status: int
    message: str


class SuccessResponse(ApiResponse, BaseModel):
    status: int = status.HTTP_200_OK
    message: str = "Successful"


class CreatedResponse(ApiResponse):
    status: int = status.HTTP_201_CREATED
    message: str = "Resource created successfully"


class NoContentResponse(ApiResponse):
    status: int = status.HTTP_204_NO_CONTENT
    message: str = "Request was executed successfully, but no content was found"


class BadRequestResponse(ApiResponse):
    status: int = status.HTTP_400_BAD_REQUEST
    message: str = "Bad request."


class ForbiddenResponse(ApiResponse):
    status: int = status.HTTP_403_FORBIDDEN
    message: str = "Request was executed, but provided keys and credentials were rejected"


class NotFoundResponse(ApiResponse):
    status: int = status.HTTP_404_NOT_FOUND
    message: str = "Resource not found"


class InternalServerErrorResponse(ApiResponse):
    status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "During execution an error on the server occurred"
    error: str = ""
