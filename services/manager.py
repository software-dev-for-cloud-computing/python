from fastapi import UploadFile, File
from fastapi.exceptions import RequestValidationError, HTTPException

from core.config import QDRANT_HOST, QDRANT_COLLECTION_NAME, OPENAI_API_KEY
from services.chain_managers.chain_manager import ChainManager
from services.rag.loader import get_pdf_text, get_chunks
from services.rag.openai_models import OpenAIModels
from services.vectore_store.document_service import DocumentService
from services.vectore_store.vector_store_service import VectorStoreService


# services/manager.py
async def upload_process(data: bytes, user_id: str, document_id: str):
    try:
        text = await get_pdf_text(data)
        chunks = get_chunks(text, document_id, owner_id=user_id, chunk_size=500)

        models = OpenAIModels(api_key=OPENAI_API_KEY)
        embedding = models.get_embedding()
        vector_store_client = VectorStoreService()
        vector_store_client.connect(collection_name=QDRANT_COLLECTION_NAME)

        document_service = DocumentService(vector_store_service=vector_store_client)
        document_service.add_documents(documents=chunks, embeddings=embedding)

        return {"message": "File uploaded successfully"}
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def search_process(question: str, user_id: str, document_id, k: int = 5):
    try:
        vector_store_client = VectorStoreService()
        vector_store_client.connect(collection_name=QDRANT_COLLECTION_NAME)

        document_service = DocumentService(vector_store_service=vector_store_client)
        result = document_service.similarity_search(query=question, owner_id=user_id, document_id=document_id, k=k)

        return result
    except RequestValidationError:
        raise HTTPException(status_code=400, detail="Invalid input")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def retrieve_process(query: str, user_id: str, conversation_id: str, document_id: str | None = None):

    models = OpenAIModels(api_key=OPENAI_API_KEY)
    llm = models.get_llm(model_name="gpt-3.5-turbo-0125")

    print("Retrieving documents...")

    vector_store_client = VectorStoreService()
    vector_store_client.connect(collection_name=QDRANT_COLLECTION_NAME)

    document_service = DocumentService(vector_store_service=vector_store_client)
    retriever = document_service.retrieve_documents(owner_id=user_id, document_id=document_id)

    chain = ChainManager(llm=llm, retriever=retriever)
    return chain.run(question=query, user_id=user_id, conversation_id=conversation_id)

