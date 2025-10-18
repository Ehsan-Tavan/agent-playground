from langchain_community.document_loaders import Docx2txtLoader
from .base_loader import BaseLoader

class DOCXLoader(BaseLoader):
    """Loader for DOCX documents."""

    def load(self, file_path):
        loader = Docx2txtLoader(file_path)
        return loader.load()
