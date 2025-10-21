import os
from langchain_openai import ChatOpenAI

from multi_agent_systems.researcher_agent.src.search_agent.prompts import get_query_generator_prompt
from multi_agent_systems.researcher_agent.src.search_agent.graph.state import SearchState

from .schema import Queries


class QueryGeneratorNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state: SearchState):
        topic = state["topic"]
        queries = self.chain.invoke({
            "topic": topic,
        })
        return {"queries": queries}


def get_query_generator_node(config: dict):
    llm = ChatOpenAI(
        model=config["llm"]["name"],
        temperature=config["llm"]["temperature"],
        api_key=os.getenv("OPENAI_API_KEY", config["llm"].get("api_key", None)),
        base_url=os.getenv("OPENAI_API_BASE", config["llm"].get("base_url", None))
    )
    structured_llm = llm.with_structured_output(Queries)

    prompt_template = get_query_generator_prompt(
        number_of_queries=config["query_generation"]["number_of_queries"]
    )

    chain = prompt_template | structured_llm

    return QueryGeneratorNode(chain=chain)
