from .hf_embedding import HFEmbedding
from .openai_embedding import OpenAIEmbedding
from .base_embedding import BaseEmbedding


class EmbeddingFactory:
    """
    Factory class for creating embedding models.

    Supported strategies:
        - "huggingface" → HFEmbedding
        - "openai" → OpenAIEmbedding

    Example:
        >>> emb = EmbeddingFactory.get_embedding("huggingface", model_path="BAAI/bge-small-en")
        >>> model = emb.get_model()
    """

    _embedding_map = {
        "huggingface": HFEmbedding,
        "openai": OpenAIEmbedding,
    }

    @staticmethod
    def get_embedding(name: str, **kwargs) -> BaseEmbedding:
        """
        Return an embedding strategy instance based on the given name.

        Args:
            name (str): The embedding backend ("huggingface" or "openai").
            **kwargs: Additional keyword arguments for the embedding class.

        Returns:
            BaseEmbedding: Instance of a subclass of BaseEmbedding.

        Raises:
            ValueError: If the embedding name is not supported.
        """
        name = name.lower()
        if name not in EmbeddingFactory._embedding_map:
            raise ValueError(f"Unsupported embedding type: {name}")
        return EmbeddingFactory._embedding_map[name](**kwargs)


# -------------------------------------------------------------------
# Example usage (for standalone testing)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1 — HuggingFace Embedding
    emb = EmbeddingFactory.get_embedding(
        "huggingface",
        model_path="/mnt/disk2/ehsan.tavan/search_env/embedding_model/Qwen3-Embedding-0.6B",
        device="cuda"
    )
    model = emb.get_model()
    print("✅ HuggingFace embedding model loaded successfully!")

    # Example 2 — OpenAI Embedding
    # emb = EmbeddingFactory.get_embedding(
    #     "openai",
    #     model_name="text-embedding-3-small",
    #     api_key="YOUR_API_KEY"
    # )
    # model = emb.get_model()
    # print("✅ OpenAI embedding model loaded successfully!")
