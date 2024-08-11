from fastapi import FastAPI

from app.api.v1 import qa, chunks, collections

app = FastAPI()

app.include_router(qa.router, prefix="/api/v1", tags=["QA"])
app.include_router(chunks.router, prefix="/api/v1", tags=["documents"])
app.include_router(collections.router, prefix="/api/v1", tags=["collections"])


