from fastapi import HTTPException, status
from app.models.dto.interfaces import InternalServerErrorResponse


class HTTPInternalServerError(HTTPException):
    def __init__(self, error: str = ""):
        self.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = InternalServerErrorResponse(
            error=error
        ).dict()
        super().__init__(status_code=self.status_code, detail=detail)
