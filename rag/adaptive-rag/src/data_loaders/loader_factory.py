import os
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .base_loader import BaseLoader


class LoaderFactory:
    """
    Factory class responsible for returning the appropriate document loader
    based on the file extension.

    This class implements the Factory Pattern, ensuring that new document
    types can be added easily without modifying existing code.

    Supported file types:
        - .html / .htm → HTMLLoader
        - .pdf → PDFLoader
        - .docx → DOCXLoader

    Example:
        >>> loader = LoaderFactory.get_loader("example.pdf")
        >>> docs = loader.load("example.pdf")
        >>> print(len(docs))
    """

    _loader_map = {
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
    }

    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext not in LoaderFactory._loader_map:
            raise ValueError(f"No loader available for file: {file_path}")
        return LoaderFactory._loader_map[ext]()


if __name__ == "__main__":
    files = [
        "../data/raptor.html",
        "../data/paper.pdf",
        "../data/report.docx"
    ]

    docs = []
    for path in files:
        loader = LoaderFactory.get_loader(path)
        docs.extend(loader.load(path))

    print(f"✅ Loaded {len(docs)} documents from {len(files)} files.")
