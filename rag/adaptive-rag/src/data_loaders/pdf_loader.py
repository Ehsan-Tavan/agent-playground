from langchain_community.document_loaders import PyPDFLoader
from .base_loader import BaseLoader

class PDFLoader(BaseLoader):
    """Loader for PDF documents."""

    def load(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()
