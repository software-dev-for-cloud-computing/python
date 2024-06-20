from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from langchain.schema import Document
from langchain.text_splitter import SpacyTextSplitter
import os
import tempfile
from fastapi import UploadFile
from aiofile import AIOFile
from langchain_community.document_loaders import PyMuPDFLoader

# services/rag/loader.py
async def get_pdf_text(data: bytes):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # Write the uploaded file data to the temporary file using aiofile
    async with AIOFile(temp_file.name, 'wb') as afp:
        await afp.write(data)
        await afp.fsync()

    # Pass the temporary file path to the PyMuPDFLoader class
    loader = PyMuPDFLoader(file_path=temp_file.name)
    data = loader.load()

    # Delete the temporary file
    os.unlink(temp_file.name)

    return data


def get_chunks(data, document_id: str, owner_id: str, chunk_size: int = 500, ):
    text_splitter = SpacyTextSplitter(chunk_size=chunk_size)
    chunks = []
    for page_number, page in enumerate(data, start=1):
        page_chunks = text_splitter.split_text(page.page_content)
        for chunk in tqdm(page_chunks):
            metadata = {"document_id": document_id, "owner_id": owner_id, "page_number": page_number}
            chunks.append(Document(page_content=chunk, metadata=metadata))
    return chunks
