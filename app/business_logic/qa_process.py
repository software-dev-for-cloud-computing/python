from fastapi.exceptions import RequestValidationError, HTTPException


class QAProcess:
    @staticmethod
    async def start_process(
            question: str,
            user_id: str,
            document_id: str,
            k: int,
            vector_store: VectorStore,
            document_service: DocumentService,
            models: OpenAIModels
    ):
        try:
            # Search for similar documents
            result = document_service.similarity_search(query=question, owner_id=user_id, document_id=document_id, k=k)
            return result
        except RequestValidationError:
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))



    ):