from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class RewriteQuestionNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        improved_question = self.chain.invoke({"question": question})
        return {"documents": documents, "question": improved_question}


def get_rewrite_question_node(
        llm_model: str,
        api_key: str,
        base_url: str,
        temperature: float
) -> RewriteQuestionNode:
    llm = ChatOpenAI(model=llm_model,
                     api_key=api_key,
                     base_url=base_url,
                     temperature=temperature)

    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter_chain = re_write_prompt | llm | StrOutputParser()
    return RewriteQuestionNode(chain=question_rewriter_chain)
