from fastapi import UploadFile

from app.interfaces.embedding_model import EmbeddingModel
from app.interfaces.pdf_reader import PDFReader
from app.interfaces.text_splitter import TextSplitter
from app.interfaces.vector_store import VectorStore
from app.models.dto.documents import UploadDocumentResponse
from app.models.dto.documents import UploadDocumentRequestBody
from app.models.dto.interfaces import InternalServerErrorResponse
from app.models.objects.pdf_upload_model import PDFUploadModel
from app.utils.logger import Logger

logger = Logger(name="Logger")


class UploadProcess:
    @staticmethod
    async def start_process(
            request_id: str,
            body: UploadDocumentRequestBody,
            file: UploadFile,
            pdf_reader: PDFReader,
            text_splitter: TextSplitter,
            embedding_model: EmbeddingModel,
            vector_store: VectorStore,
            document_id: str,
            owner_id: str
    ) -> UploadDocumentResponse | InternalServerErrorResponse:
        try:
            # Read PDF content from the file in the request
            pdf_content = await file.read()

            # Convert request into request model
            logger.log(level="debug", func_name="upload_process", message="Convert request into request model")
            pdf_data = PDFUploadModel(file=pdf_content)

            # Read PDF content
            pdf_text = await pdf_reader.read_pdf(pdf_data.file)

            # Chunk the text
            chunks = text_splitter.split_text(text=pdf_text, document_id=document_id, owner_id=owner_id)

            # Estimate token usage for embedding
            logger.log(level="debug", func_name="upload_process", message="Get number of tokens for the embedding")
            tokens = embedding_model.estimate_tokens_chunks(chunks)

            # Save to Qdrant
            logger.log(level="debug", func_name="upload_process",
                       message="Add the newly created chunks to the vectorstore")
            chunks_added = vector_store.add_chunks(chunks=chunks, embedding_model=embedding_model)

            print(chunks_added)

            # Return response
            return UploadDocumentResponse(
                requestId=request_id,
                documentId=body.document_id,
                ownerId=body.owner_id,
                totalNumberOfChunksCreated=len(chunks),
                totalNumberOfTokensUsed=tokens
            )

        except Exception as e:
            return InternalServerErrorResponse(message=str(e))
