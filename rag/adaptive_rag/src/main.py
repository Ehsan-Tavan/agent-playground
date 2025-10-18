from data_loaders import LoaderFactory
from embeddings import EmbeddingFactory
from retriever import TextSplitter
from retriever import RetrieverCore


def retrieve(state, retriever):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)

        return {"documents": documents, "question": question}

def main():
    local_files = [
        "../data/raptor.html",
        "../data/biomedical_rag.html",
        "../data/llamaindex_comparison.html",
        "../data/multihead_attention.html",
        "../data/transformer_embeddings.html"
    ]

    docs = []
    for path in local_files:
        loader = LoaderFactory.get_loader(path)
        docs.extend(loader.load(path))

    print(f"✅ Loaded {len(docs)} documents from {len(local_files)} files.")

    embedding_strategy = EmbeddingFactory.get_embedding(
        name="huggingface",
        model_path="/mnt/disk2/ehsan.tavan/search_env/embedding_model/Qwen3-Embedding-0.6B",
        device="cuda"
    )

    embedding_model = embedding_strategy.get_model()

    doc_splits = TextSplitter().split(docs)

    texts = [doc.page_content for doc in doc_splits]
    metadatas = [doc.metadata for doc in doc_splits]

    retriever_core = RetrieverCore(embedding_model=embedding_model)
    retriever_core.add_documents(texts=texts, metadatas=metadatas)
    retriever = retriever_core.get_retriever()

    retriever_output = retrieve(state={"question": "What is RAPTOR?"}, retriever=retriever)
    print(retriever_output)

if __name__ == "__main__":
    main()