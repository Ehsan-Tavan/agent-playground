from langchain_community.document_loaders import BSHTMLLoader
from .base_loader import BaseLoader

class HTMLLoader(BaseLoader):
    """Loader for HTML files using BeautifulSoup (LangChain BSHTMLLoader)."""

    def load(self, file_path):
        loader = BSHTMLLoader(file_path)
        return loader.load()
