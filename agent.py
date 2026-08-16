from datetime import date

import streamlit as st
from langchain.agents import initialize_agent, AgentType

from langchain_groq import ChatGroq

from tools import create_pdf_tool, create_web_tool, create_flight_tool


def create_agent(retriever):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    pdf_tool = create_pdf_tool(retriever)
    web_tool = create_web_tool(st.secrets["TAVILY_API_KEY"])
    flight_tool = create_flight_tool(token=st.secrets["TRAVELPAYOUTS_TOKEN"])

    # Give the agent a sense of "today" so it can resolve relative dates
    # like "this month" / "next week" into the YYYY-MM or YYYY-MM-DD
    # format the Flight Search tool actually needs. Without this the
    # agent has no way to know what "this month" means and can loop
    # indefinitely trying to figure it out.
    today = date.today().isoformat()

    system_prefix = f"""You are Concierge, an expert AI travel assistant for
India, built by Team ElevateX. Today's date is {today}.

LANGUAGE: Always reply in the SAME language and script the user just wrote
in. If they write in Hindi (Devanagari), reply in Hindi. If they write in
Hinglish (Hindi words in Roman script, e.g. "Mumbai se Goa jane ka kharcha
kitna hoga"), reply in the same natural Hinglish register - don't force it
into pure English or pure formal Hindi. If they mix English with a regional
Indian language, mirror that mix. If the language is unclear, default to
English. Never mention that you are "detecting" or "switching" languages -
just respond naturally in it, the way a multilingual Indian travel agent
would.

REASONING: Be genuinely helpful, not just literal. If a request is vague
(e.g. "flights from Mumbai"), ask one short, specific clarifying question
instead of giving up. When a user gives a relative date like "this month",
"next month", "is weekend", or "next week", convert it to an explicit
YYYY-MM or YYYY-MM-DD value yourself before calling any tool - never pass a
relative phrase directly as a tool argument. Use conversation history (given
above the latest message, if present) to keep context across turns - e.g. if
they already told you their destination, don't ask again.

TOOLS: Use the PDF Travel Guide tool for background on Indian destinations,
Web Search for anything current, and Flight Search for cached flight price
data. The Flight Search tool is backed by the same shared flight-search
function used by the Generate Travel Plan button, so chat and form searches
must behave consistently. Always mention when flight prices are cached
estimates, not live bookable fares. When a user provides a follow-up date,
combine it with the origin/destination from the conversation before calling
Flight Search.
"""

    # STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION (rather than the plain
    # ZERO_SHOT_REACT_DESCRIPTION agent) is required here because the
    # Flight Search tool takes multiple named arguments (origin,
    # destination, departure_date) instead of a single string.
    agent = initialize_agent(
        tools=[pdf_tool, web_tool, flight_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": system_prefix},
        # Safety net: if the agent can't converge on a clean "Final
        # Answer" within max_iterations, force it to generate its best
        # answer from whatever it has so far, instead of silently
        # returning an empty string.
        max_iterations=10,
        early_stopping_method="generate",
    )

    return agent
