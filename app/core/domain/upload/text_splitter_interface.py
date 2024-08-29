from abc import ABC, abstractmethod
from typing import List

from app.core.domain.chunks.chunk_model import ChunkModel


class TextSplitter(ABC):
    @abstractmethod
    def split_text(self, text, document_id: str, owner_id: str, conversation_id: str) -> List[ChunkModel]:
        pass
