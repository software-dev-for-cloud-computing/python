from abc import ABC, abstractmethod
from typing import List

from app.models.objects.chunk_model import ChunkModel


class TextSplitter(ABC):
    @abstractmethod
    def split_text(self, text, document_id: str, owner_id: str) -> List[ChunkModel]:
        pass
