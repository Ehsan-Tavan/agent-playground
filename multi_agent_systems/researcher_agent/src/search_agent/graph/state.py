from typing_extensions import TypedDict
from multi_agent_systems.researcher_agent.src.search_agent.nodes.schema import Queries


class SearchState(TypedDict):
    topic: str
    queries: Queries
