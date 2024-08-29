from abc import ABC, abstractmethod
from io import BytesIO


class PDFReader(ABC):
    @abstractmethod
    def read_pdf(self, file):  # -> str:
        pass
