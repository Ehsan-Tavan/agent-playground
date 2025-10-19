import os
import yaml
from dotenv import load_dotenv

from rag.adaptive_rag.src.data_loaders import LoaderFactory
from rag.adaptive_rag.src.embeddings import EmbeddingFactory
from rag.adaptive_rag.src.retriever import TextSplitter, RetrieverCore
from rag.adaptive_rag.src.graph import create_graph


def build_app(config_path: str):
    """Initialize and return the LangGraph app based on the config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

        # Step 1: Load documents
        docs = []
        for path in config["knowledge_bae"]["data_path"]:
            loader = LoaderFactory.get_loader(path)
            docs.extend(loader.load(path))
        print(f"✅ Loaded {len(docs)} documents from {len(config['knowledge_bae']['data_path'])} files.")

        # Step 2: Embedding setup
        embedding_strategy = EmbeddingFactory.get_embedding(
            name=config["embedding_model"]["name"],
            model_path=config["embedding_model"]["model_path"],
            device=config["embedding_model"]["device"],
        )
        embedding_model = embedding_strategy.get_model()

        # Step 3: Split documents
        doc_splits = TextSplitter().split(docs)
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [doc.metadata for doc in doc_splits]

        # Step 4: Create retriever
        retriever_core = RetrieverCore(embedding_model=embedding_model)
        retriever_core.add_documents(texts=texts, metadatas=metadatas)
        retriever = retriever_core.get_retriever()

        # Step 5: Build LangGraph app
        app = create_graph(retriever=retriever, config=config["llm"])
        return app

def format_response(result):
    """Extract and format response from workflow result."""
    if isinstance(result, dict):
        if "generation" in result:
            return result["generation"]
        if "answer" in result:
            return result["answer"]
    return str(result)


def main():
    """Interactive CLI for Adaptive RAG System."""
    print("🌐 Adaptive RAG System (LangGraph CLI)")
    print("Type 'quit' or 'exit' to end.\n")

    config_path = os.getenv("CONFIG_PATH")
    if not config_path:
        raise EnvironmentError(
            "Environment variable 'CONFIG_PATH' is not set. Please define it in your .env file or shell.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    app = build_app(config_path)

    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ["quit", "exit", "q", ""]:
                print("👋 Exiting...")
                break

            result = None
            for output in app.stream({"question": question}):
                for key, value in output.items():
                    result = value

            if result:
                print(f"\n💡 Answer: {format_response(result)}\n")
            else:
                print("⚠️ No response generated.\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break

        except Exception as e:
            print(f"Error: {str(e)}\n")
