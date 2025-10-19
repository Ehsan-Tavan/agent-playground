from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document


def get_web_search_node(k):
    web_search_tool = TavilySearchResults(k=3)

    def _web_search(state):
        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    return _web_search
