from fastapi import UploadFile, File
from pydantic import BaseModel


class UploadFileModel(BaseModel):
    userId: str
    conversationId: str
    file: UploadFile = File(...)
