from typing import List
from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[Query] =  Field(description="List of web search queries.")
