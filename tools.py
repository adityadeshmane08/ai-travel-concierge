from langchain.tools import Tool
from tavily import TavilyClient

# PDF Tool
def create_pdf_tool(retriever):
    return Tool(
        name="PDF Travel Guide",
        description="Use this tool to answer questions from the India travel guide PDF.",
        func=lambda question: "\n".join(
            doc.page_content for doc in retriever.invoke(question)
        ),
    )

# Tavily Web Search Tool
def create_web_tool(api_key):
    tavily = TavilyClient(api_key=api_key)

    return Tool(
        name="Web Search",
        description="Use this tool for live travel information, current events, weather, or anything not available in the PDF.",
        func=lambda question: str(
            tavily.search(query=question, max_results=3)
        ),
    )
