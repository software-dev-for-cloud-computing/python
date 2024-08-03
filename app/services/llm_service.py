from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator
from app.interfaces.llm_model import LlmModel
import os

load_dotenv()

print(os.getenv("LLM_MODEL"))
print(os.getenv("LLM_DEFAULT_TEMP"))
print(os.getenv("LLM_DEFAULT_TOKEN_LIMIT"))
print(os.getenv("LLM_MIN_TOKEN_LIMIT"))
print(os.getenv("LLM_MAX_TOKEN_LIMIT"))



class OpenAILLMModel(LlmModel):

    def __init__(self, api_key: str):
        self.model_name = os.getenv("LLM_MODEL")
        self.api_key = api_key
        self.temperature = float(os.getenv("LLM_DEFAULT_TEMP"))
        self.max_tokens = int(os.getenv("LLM_DEFAULT_TOKEN_LIMIT"))


    @validator('api_key')
    def validate_api_key(cls, value):
        if not value:
            raise ValueError(
                "Did not find openai_api_key, please add an environment variable `OPENAI_API_KEY` which contains it, "
                "or pass `openai_api_key` as a named parameter.")
        return value

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=True,
        )

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
