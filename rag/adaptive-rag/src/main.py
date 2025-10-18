from tqdm import tqdm

from data_loaders import LoaderFactory
from embeddings import EmbeddingFactory
from retriever import TextSplitter
from retriever import ChromaVectorStore


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

    chroma_db = ChromaVectorStore(embedding_function=embedding_model)

    chroma_db.add_documents(texts, metadatas)

    retriever = chroma_db.as_retriever()

    retriever_output = retrieve(state={"question": "What is RAPTOR?"}, retriever=retriever)
    print(retriever_output)

if __name__ == "__main__":
    main()

    # docs_list = [item for sublist in docs for item in sublist]

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=50,
    #     length_function=len
    # )
    # doc_splits = text_splitter.split_documents(docs_list)
    #
    # texts = [doc.page_content for doc in doc_splits]
    # metadatas = [doc.metadata for doc in doc_splits]
    #
    # vectorstore = Chroma(
    #     collection_name="tuhin-blogs",
    #     embedding_function=embd,
    # )
    #
    # # Add documents with progress bar
    # for i in tqdm(range(0, len(texts), 32), desc="Embedding documents"):
    #     batch_texts = texts[i:i + 32]
    #     batch_metas = metadatas[i:i + 32]
    #     vectorstore.add_texts(batch_texts, metadatas=batch_metas)
    #
    # print("done")
    #
    # retriever = vectorstore.as_retriever()
    #
    #
    # def retrieve(state):
    #     """
    #     Retrieve documents
    #
    #     Args:
    #         state (dict): The current graph state
    #
    #     Returns:
    #         state (dict): New key added to state, documents, that contains retrieved documents
    #     """
    #     print("---RETRIEVE---")
    #     question = state["question"]
    #     print(f"question {question}")
    #
    #     # Retrieval
    #     documents = retriever.invoke(question)
    #     print(f"documents {documents}")
    #
    #     return {"documents": documents, "question": question}
    #
    #
    # state = {"question": "What is RAPTOR?"}
    # a = retrieve(state)
    # print(a)
    #
    # # vectorstore = Chroma.from_documents(
    # #     documents=doc_splits,
    # #     collection_name="tuhin-blogs",
    # #     embedding=embd,
    # # )
