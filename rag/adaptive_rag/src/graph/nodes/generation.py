from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser



class GenerationNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        generation = self.chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

def get_generation_node(
        llm_model: str,
        api_key: str,
        base_url: str,
        temperature: float
) -> GenerationNode:
    llm = ChatOpenAI(model=llm_model,
                     api_key=api_key,
                     base_url=base_url,
                     temperature=temperature)

    system_prompt = """YYou are an assistant for question-answering tasks.
     Use the provided context to answer the question.
     If you don't know the answer, just say you don't know.
     Keep the answer concise and factual."""

    generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context: {context}\n\nQuestion: {question}"),
        ]
    )

    generation_chain = generation_prompt | llm | StrOutputParser()

    return GenerationNode(chain=generation_chain)
