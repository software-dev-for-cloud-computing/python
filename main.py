# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.api.v1 import qa, chunks, collections
from app.exceptions.http_exceptions import HTTPInternalServerError

app = FastAPI()


@app.exception_handler(HTTPInternalServerError)
async def http_500_error_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


app.include_router(qa.router, prefix="/api/v1", tags=["QA"])
app.include_router(chunks.router, prefix="/api/v1", tags=["documents"])
app.include_router(collections.router, prefix="/api/v1", tags=["collections"])
