from langchain_core.prompts import ChatPromptTemplate

REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic
                      and building a comprehensive report in markdown using the following format:

                      1. Introduction (no web search needed)
                            - Brief overview of the topic area

                      2. Main Body Sections:
                            - Each section should focus on a sub-topic of the user-provided topic
                            - Include any key concepts and definitions
                            - Provide real-world examples or case studies where applicable

                      3. Conclusion (no web search needed)
                            - Aim for 1 structural element (either a list of table) that distills the main body sections
                            - Provide a concise summary of the report

                      When generating the final response in markdown, if there are special characters in the text,
                      such as the dollar symbol, ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5
                  """


def get_query_generator_prompt(number_of_queries: int = 5):
    return ChatPromptTemplate(
        [
            ("system", "You are an expert technical report writer, helping to plan a comprehensive report."
                       f"Your goal is to generate {number_of_queries} search queries that will help gather "
                       "comprehensive information for planning the report sections.\n\n"
                       "The report structure will follow these guidelines:\n"
                       f"{REPORT_STRUCTURE}\n\n"
                       "**Guidelines for Generating Queries:**\n"
                       "1. Ensure each query is directly related to the topic.\n"
                       "2. Tailor the queries to satisfy the requirements outlined in the report structure.\n"
                       "3. Make the query specific enough to find high-quality, relevant sources while covering the "
                       "depth and breadth needed for the report structure.\n"
             ),
            ("user",
             "Report Topic:\n{topic}\n\n"
             "Generate a list of search queries that will help gather information for a detailed and well-structured "
             "report on this topic.")
        ]
    )
