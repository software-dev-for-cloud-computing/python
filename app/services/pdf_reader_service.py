import os
import tempfile
import time
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
        temp_filename = temp_file.name
        try:
            async with AIOFile(temp_filename, 'wb') as afp:
                await afp.write(file)
                await afp.fsync()

            # Lade die Datei in den Speicher
            loader = PyMuPDFLoader(file_path=temp_filename)
            data = loader.load()

        finally:
            # Warten, bis die Datei nicht mehr verwendet wird
            time.sleep(0.1)
            os.unlink(temp_filename)

        return data
