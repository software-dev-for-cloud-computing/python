from fastapi import UploadFile

from app.core.domain.chunks.chunk_repository import ChunkRepository
from app.core.domain.upload.pdf_reader_interface import PDFReader
from app.core.domain.upload.text_splitter_interface import TextSplitter
from app.exceptions.http_exceptions import HTTPInternalServerError
from app.core.external_services.embedding.embedding_port import EmbeddingModel
from app.models.dto.documents import UploadDocumentResponse
from app.models.dto.documents import UploadDocumentRequest
from app.models.dto.interfaces import InternalServerErrorResponse
from app.models.objects.pdf_upload_model import PDFUploadModel
from app.core.utils.logger import Logger

logger = Logger(name="Logger")


class UploadProcess:
    @staticmethod
    async def start_process(
            request_id: str,
            params: UploadDocumentRequest,
            file: UploadFile,
            pdf_reader: PDFReader,
            text_splitter: TextSplitter,
            embedding_model: EmbeddingModel,
            chunk_repository: ChunkRepository,
            document_id: str,
            owner_id: str
    ) -> UploadDocumentResponse | InternalServerErrorResponse:

        # Read PDF content from the file in the request
        pdf_content = await file.read()

        # Convert request into request model
        logger.log(level="debug", func_name="upload_process", message="Convert request into request model")
        pdf_data = PDFUploadModel(file=pdf_content)

        # Read PDF content
        pdf_text = await pdf_reader.read_pdf(pdf_data.file)

        # Chunk the text
        chunks = text_splitter.split_text(text=pdf_text,
                                          document_id=document_id,
                                          owner_id=owner_id,
                                          conversation_id=params.conversation_id)

        # Estimate token usage for embedding
        logger.log(level="debug", func_name="upload_process", message="Get number of tokens for the embedding")
        tokens = embedding_model.estimate_tokens_chunks(chunks)

        # Save to Qdrant
        logger.log(level="debug", func_name="upload_process",
                   message="Add the newly created chunks to the vectorstore")
        chunks_added = chunk_repository.add_chunks(chunks=chunks, embedding_model=embedding_model)

        # Check if the chunks were added successfully
        if not chunks_added:
            e = Exception("Internal server error: Could not add chunks to the vector store")
            raise HTTPInternalServerError(error=str(e))

        # Return response
        return UploadDocumentResponse(
            requestId=request_id,
            documentId=params.document_id,
            ownerId=params.owner_id,
            totalNumberOfChunksCreated=len(chunks),
            totalNumberOfTokensUsed=tokens
        )
