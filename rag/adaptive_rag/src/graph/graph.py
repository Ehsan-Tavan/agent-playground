from langgraph.graph import StateGraph, START, END
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

from rag.adaptive_rag.src.graph.nodes import (get_web_search_node, get_retriever_node,
                                              get_retrieval_grader_node, get_generation_node,
                                              get_rewrite_question_node, get_question_router_node,
                                              get_generation_decider_node, get_answer_grader_node)

from .state import GraphState


def create_graph(retriever: VectorStoreRetriever, config: dict):
    workflow = StateGraph(GraphState)

    web_search_node = get_web_search_node(k=3)
    retrieve_node = get_retriever_node(retriever=retriever)
    grade_documents_node = get_retrieval_grader_node(
        llm_model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"]
    )
    generate_node = get_generation_node(
        llm_model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"]
    )

    transform_query_node = get_rewrite_question_node(
        llm_model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"]
    )

    question_router = get_question_router_node(
        llm_model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"]
    )

    generation_decider = get_generation_decider_node()
    answer_grader = get_answer_grader_node(
        llm_model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"]
    )

    workflow.add_node("web_search", web_search_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("transform_query", transform_query_node)

    workflow.add_conditional_edges(
        START,
        question_router,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        generation_decider,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "transform_query",
        question_router,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_conditional_edges(
        "generate",
        answer_grader,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    app = workflow.compile()

    # plot = app.get_graph(xray=True).draw_mermaid_png(
    #     draw_method=MermaidDrawMethod.PYPPETEER,
    # )
    # with open("../images/adaptive_rag.png", "wb") as fp:
    #     fp.write(plot)

    return app
