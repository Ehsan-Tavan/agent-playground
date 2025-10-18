from langchain_openai import OpenAIEmbeddings
from .base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding strategy."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    def get_model(self):
        """
        Initialize and return an OpenAI embedding model.
        """
        return OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key
        )
