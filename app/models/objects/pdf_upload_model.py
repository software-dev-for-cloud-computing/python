from pydantic import BaseModel, field_validator


class PDFUploadModel(BaseModel):
    file: bytes

    @field_validator('file', mode='before')
    def check_file_type(cls, v):
        if not v.startswith(b"%PDF"):
            raise ValueError("Invalid file type. Please upload a PDF file.")
        return v