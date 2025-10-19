from typing import Dict, List
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents import Document


class RetrieverNode:
    def __init__(self, retriever: VectorStoreRetriever):
        print("\n==== RETRIEVE ====\n")
        self.retriever = retriever

    def __call__(self, state) -> Dict[str, List[Document]]:
        return {
            "documents": self.retriever.invoke(state["question"])
        }


def get_retriever_node(retriever: VectorStoreRetriever) -> RetrieverNode:
    return RetrieverNode(retriever=retriever)
