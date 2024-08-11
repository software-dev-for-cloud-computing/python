import os
from typing import List

import tiktoken
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import PositiveInt

from app.interfaces.embedding_model import EmbeddingModel
from app.models.objects.chunk_model import ChunkModel
from app.utils.logger import Logger

logger = Logger('Logger')

load_dotenv()

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL")
DEFAULT_DIMENSIONS = os.getenv("VECTOR_STORE_DIMENSION")


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL, dimensions: PositiveInt = DEFAULT_DIMENSIONS):
        self.api_key = api_key
        self.model_name: str = model_name
        self.dimensions: PositiveInt = dimensions
        self.model = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key,
            dimensions=self.dimensions,
            show_progress_bar=True
        )
        self.tokenizer = tiktoken.get_encoding('cl100k_base')  # Example for the `gpt-3.5-turbo` model

    def get_model(self):
        return self.model

    def set_model(self, model_name: str):
        self.model.model = model_name

    def set_dimensions(self, dimensions: PositiveInt):
        self.model.dimensions = dimensions

    def estimate_tokens_text(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def estimate_tokens_chunks(self, chunks: List[ChunkModel]) -> int:
        return sum(self.estimate_tokens_text(chunk.content) for chunk in chunks)


