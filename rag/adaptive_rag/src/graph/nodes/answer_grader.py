from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class AnswerGraderNode:
    def __init__(self, hallucination_grader_chain, answer_grader_chain):
        self.hallucination_grader_chain = hallucination_grader_chain
        self.answer_grader_chain = answer_grader_chain

    def __call__(self, state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        hallucination_score = self.hallucination_grader_chain.invoke(
            {"documents": documents, "generation": generation}
        )
        hallucination_grade = hallucination_score.binary_score

        # Check hallucination
        if hallucination_grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            answer_grader_score = self.answer_grader_chain.invoke({"question": question, "generation": generation})
            answer_grader_grade = answer_grader_score.binary_score
            if answer_grader_grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"


def get_answer_grader_node(
        llm_model: str,
        api_key: str,
        base_url: str,
        temperature: float
) -> AnswerGraderNode:
    llm = ChatOpenAI(model=llm_model,
                     api_key=api_key,
                     base_url=base_url,
                     temperature=temperature)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader_chain = hallucination_prompt | structured_llm_grader

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader_chain = answer_prompt | structured_llm_grader

    return AnswerGraderNode(hallucination_grader_chain=hallucination_grader_chain,
                            answer_grader_chain=answer_grader_chain)




