from typing import List
from langchain_text_splitters import SpacyTextSplitter
from tqdm import tqdm

from app.interfaces.text_splitter import TextSplitter
from app.models.objects.chunk_model import ChunkModel, ChunkMetadata
from app.utils.logger import Logger

logger = Logger('Logger')


class TextSplitterService(TextSplitter):
    def __init__(self, chunk_size: int):
        self.splitter = SpacyTextSplitter(chunk_size=chunk_size)

    @logger.log_decorator(level="debug", message="Creating chunks")
    def split_text(self, text, document_id: str, owner_id: str) -> List[ChunkModel]:
        chunks = []
        for page_number, page in enumerate(text, start=1):
            page_chunks = self.splitter.split_text(page.page_content)
            for chunk in tqdm(page_chunks):
                metadata = ChunkMetadata(document_id=document_id, owner_id=owner_id, page_number=page_number)
                chunk_obj = ChunkModel(content=chunk, metadata=metadata)
                chunks.append(chunk_obj)
        return chunks
