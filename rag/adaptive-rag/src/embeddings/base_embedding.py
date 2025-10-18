from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract base class for embedding strategies."""

    @abstractmethod
    def get_model(self):
        """Return an initialized embedding model."""
        pass
