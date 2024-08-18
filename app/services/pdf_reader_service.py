import os
import tempfile
from aiofile import AIOFile
from langchain_community.document_loaders import PyMuPDFLoader

from app.interfaces.pdf_reader import PDFReader
from app.utils.logger import Logger

logger = Logger('Logger')


class PDFReaderService(PDFReader):
    @staticmethod
    @logger.log_decorator(level="debug", message="Reading in PDF")
    async def read_pdf(file):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            async with AIOFile(temp_file.name, 'wb') as afp:
                await afp.write(file)
                await afp.fsync()
            loader = PyMuPDFLoader(file_path=temp_file.name)
            data = loader.load()
        finally:
            os.unlink(temp_file.name)
        return data