import os
from tqdm import tqdm
from langchain_community.vectorstores import Chroma


class ChromaVectorStore:
    """Chroma implementation of a vector store (Repository Pattern)"""

    def __init__(self, embedding_function, persist_dir="./vector_db", collection_name="tuhin-blogs"):
        os.makedirs(persist_dir, exist_ok=True)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_dir
        )

    def add_documents(self, texts, metadatas, batch_size=32):
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            self.db.add_texts(texts[i:i + batch_size], metadatas=metadatas[i:i + batch_size])
        self.db.persist()

    def as_retriever(self):
        return self.db.as_retriever()
