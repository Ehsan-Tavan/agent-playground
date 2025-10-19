import os
import yaml
import argparse

from rag.adaptive_rag.src.data_loaders import LoaderFactory
from rag.adaptive_rag.src.embeddings import EmbeddingFactory
from rag.adaptive_rag.src.retriever import TextSplitter, RetrieverCore
from rag.adaptive_rag.src.graph import create_graph


def main(config):
    docs = []
    for path in config["knowledge_bae"]["data_path"]:
        loader = LoaderFactory.get_loader(path)
        docs.extend(loader.load(path))

    print(f"✅ Loaded {len(docs)} documents from {len(config['knowledge_bae']['data_path'])} files.")

    embedding_strategy = EmbeddingFactory.get_embedding(
        name=config["embedding_model"]["name"],
        model_path=config["embedding_model"]["model_path"],
        device=config["embedding_model"]["device"],
    )

    embedding_model = embedding_strategy.get_model()

    doc_splits = TextSplitter().split(docs)

    texts = [doc.page_content for doc in doc_splits]
    metadatas = [doc.metadata for doc in doc_splits]

    retriever_core = RetrieverCore(embedding_model=embedding_model)
    retriever_core.add_documents(texts=texts, metadatas=metadatas)
    retriever = retriever_core.get_retriever()

    graph = create_graph(retriever=retriever, config=config["llm"])

    # Run
    inputs = {
        "question": "Building a Biomedical Question-Answering System Using RAG"
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
        print("\n---\n")

    # Final generation
    print(value["generation"])

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Adaptive RAG"
    )
    PARSER.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help="Config file path (default: None)"
    )

    ARGS = PARSER.parse_args()

    CONFIG = yaml.safe_load(open(ARGS.config))

    main(config=CONFIG)