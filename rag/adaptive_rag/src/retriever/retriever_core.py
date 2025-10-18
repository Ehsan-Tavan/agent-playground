from typing import List

from rag.adaptive_rag.src.embeddings.base_embedding import BaseEmbedding

from .chroma_store import ChromaVectorStore

class RetrieverCore:
    def __init__(self, embedding_model: BaseEmbedding):
        self.chroma_db = ChromaVectorStore(embedding_function=embedding_model)

    def add_documents(self, texts: List[str], metadatas: List[str]):

        self.chroma_db.add_documents(texts, metadatas)

    def get_retriever(self):
        return self.chroma_db.as_retriever()
