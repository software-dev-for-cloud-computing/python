import os

from dotenv import load_dotenv

load_dotenv()

# QDRANT SETTINGS
QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "CoStudy"

# OPENAPI SETTINGS
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
