from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGraderNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.chain.invoke(
                {"question": question, "document": d.page_content}
            )

            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


def get_retrieval_grader_node(
        llm_model: str,
        api_key: str,
        base_url: str,
        temperature: int
) -> RetrievalGraderNode:
    llm = ChatOpenAI(model=llm_model,
                     api_key=api_key,
                     base_url=base_url,
                     temperature=temperature)

    structured_llm_grader = llm.with_structured_output(GradeDocuments, method="function_calling")

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""


    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader_chain = grade_prompt | structured_llm_grader

    return RetrievalGraderNode(chain=retrieval_grader_chain)


