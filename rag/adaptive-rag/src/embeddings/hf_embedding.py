from langchain_huggingface import HuggingFaceEmbeddings
from .base_embedding import BaseEmbedding


class HFEmbedding(BaseEmbedding):
    """HuggingFace embedding strategy."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device

    def get_model(self):
        """
        Initialize and return a HuggingFace embedding model.
        """
        return HuggingFaceEmbeddings(
            model_name=self.model_path,
            model_kwargs={"device": self.device}
        )
