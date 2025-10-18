from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """Load documents from a file path."""
        pass
