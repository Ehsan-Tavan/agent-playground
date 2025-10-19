from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class QuestionRouterNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state) -> str:
        print("---ROUTE Question---")
        source = self.chain.invoke({"question": state["question"]})

        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            raise NotImplementedError(source.datasource)


def get_question_router_node(
        llm_model: str,
        api_key: str,
        base_url: str,
        temperature: float
) -> QuestionRouterNode:
    llm = ChatOpenAI(model=llm_model,
                     api_key=api_key,
                     base_url=base_url,
                     temperature=temperature)
    structured_llm_router = llm.with_structured_output(RouteQuery, method="function_calling")

    system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to RAPTOR, llamastack,llamaindex, langchain,haystack and transformer architecture.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    question_router_chain = route_prompt | structured_llm_router

    return QuestionRouterNode(chain=question_router_chain)
