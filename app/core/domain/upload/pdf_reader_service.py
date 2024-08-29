import os
import tempfile
import time
from aiofile import AIOFile
from langchain_community.document_loaders import PyMuPDFLoader

from app.core.domain.upload.pdf_reader_interface import PDFReader
from app.core.utils.logger import Logger

logger = Logger('Logger')


class PDFReaderService(PDFReader):
    @staticmethod
    @logger.log_decorator(level="debug", message="Reading in PDF")
    async def read_pdf(file):
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = temp_file.name
        try:
            async with AIOFile(temp_filename, 'wb') as afp:
                await afp.write(file)
                await afp.fsync()

            # Read in PDF
            loader = PyMuPDFLoader(file_path=temp_filename)
            data = loader.load()

            # Delete resources
            del loader

        finally:
            # Ensure file is not blocked by the os
            time.sleep(0.1)
            os.unlink(temp_filename)

        # return data
        return data
