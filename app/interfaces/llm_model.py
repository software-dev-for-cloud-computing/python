import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from pydantic import Field

load_dotenv()

class LlmModel(ABC):
    model_name: str = Field(os.getenv("LLM_MODEL"))

    api_key: str = Field(min_length=30)

    temperature: float = Field(os.getenv("LLM_DEFAULT_TEMP"),
                               ge=os.getenv("LLL_MIN_TEMP"),
                               le=os.getenv("LLM_MAX_TEMP"))

    max_tokens: int = Field(os.getenv("LLM_DEFAULT_TOKEN_LIMIT"),
                            ge=os.getenv("LLM_MIN_TOKEN_LIMIT"),
                            le=os.getenv("LLM_MAX_TOKEN_LIMIT"))

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
