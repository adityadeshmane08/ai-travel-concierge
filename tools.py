from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

import travelpayouts_client

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


# Travelpayouts Flight Search Tool
# (Amadeus's self-service portal was shut down on 2026-07-17, so this
# project uses Travelpayouts' free Data API instead - see
# travelpayouts_client.py for signup instructions.)
class FlightSearchInput(BaseModel):
    origin: str = Field(description="Departure city or airport, e.g. 'Mumbai' or 'BOM'")
    destination: str = Field(description="Arrival city or airport, e.g. 'Goa' or 'GOI'")
    departure_date: str = Field(description="Departure month or date, YYYY-MM or YYYY-MM-DD")


def create_flight_tool(token: str):
    def _search(origin: str, destination: str, departure_date: str) -> str:
        try:
            offers = travelpayouts_client.search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                token=token,
            )
        except Exception as exc:
            return f"Flight search failed: {exc}"

        if not offers:
            return "No cached flight price data found for that route/date."

        lines = []
        for o in offers:
            lines.append(
                f"{o['airline']}{o['flight_number']} | {o['origin']} -> {o['destination']} | "
                f"departs {o['departure_at']} | {o['price']} {o['currency']}"
            )
        return (
            "Note: prices are cached recent searches, not a live quote.\n"
            + "\n".join(lines)
        )

    return StructuredTool.from_function(
        func=_search,
        name="Flight Search",
        description=(
            "Use this tool to find recent cached flight prices between two "
            "cities for a given month/date. Always call this when the user "
            "wants flight options or price estimates."
        ),
        args_schema=FlightSearchInput,
    )
