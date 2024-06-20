import os
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.singleton import SingletonMeta


class OpenAIModels(metaclass=SingletonMeta):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None
        self.embedding = None

    def get_llm(self, model_name: str = "gpt-3.5-turbo-0125"):
        if self.llm is None:
            self.llm = ChatOpenAI(model=model_name,
                                  api_key=self.api_key,
                                  temperature=0,  # Default 0,7, this is a more conservative value
                                  max_tokens=180,  # Sets a maximum number of tokens to generate
                                  verbose=True,  # Returns log output
                                  )
        return self.llm

    def get_embedding(self, model_name: str = "text-embedding-3-small"):
        if self.embedding is None:
            self.embedding = OpenAIEmbeddings(model=model_name,
                                              openai_api_key=self.api_key,
                                              show_progress_bar=True)

        return self.embedding