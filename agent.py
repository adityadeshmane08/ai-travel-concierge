from datetime import date

import streamlit as st
from langchain.agents import initialize_agent, AgentType

from langchain_groq import ChatGroq

from tools import create_pdf_tool, create_web_tool, create_flight_tool, create_weather_tool


def create_agent(retriever):
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    pdf_tool = create_pdf_tool(retriever)
    web_tool = create_web_tool(st.secrets["TAVILY_API_KEY"])
    flight_tool = create_flight_tool(token=st.secrets["TRAVELPAYOUTS_TOKEN"])
    weather_tool = create_weather_tool()

    today = date.today().isoformat()

    system_prefix = f"""You are Concierge, an expert AI travel assistant for
India. Today's date is {today}.

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
Web Search for anything current that isn't covered by a dedicated tool,
Weather Forecast for real current conditions and short-term forecasts for a
city, and Flight Search for cached flight price data. The Flight Search
tool is backed by the same shared flight-search function used by the
Generate Travel Plan button, so chat and form searches must behave
consistently. When a user provides a follow-up date, combine it with the
origin/destination from the conversation before calling Flight Search.
Always prefer the Weather Forecast tool over Web Search for weather
questions - it returns real structured data instead of a guess.
CRITICAL FORMATTING INSTRUCTION: 
When providing your action, you must output raw JSON only. Do NOT wrap your JSON response in ```json ... ``` markdown formatting blocks. For simple conversational questions that don't need a tool (e.g. "what's the best place to visit in Pune", general recommendations, opinions), you may still use the PDF Travel Guide or Web Search tool to ground your answer, but always finish with a single clean Final Answer - never leave your reasoning half-written or stop mid-thought.
"""

    agent = initialize_agent(
        tools=[pdf_tool, web_tool, flight_tool, weather_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": system_prefix},
        max_iterations=10,
        early_stopping_method="generate",
    )

    return agent


def create_fallback_llm():
    """
    A plain (no tools, no structured-output format) LLM used only as a
    last resort when the main agent's structured JSON output fails to
    parse. Talking to the model directly, without forcing the ReAct/JSON
    action format, almost always produces a clean answer even when the
    agent path fails, so the user gets a real answer instead of a raw
    error.
    """
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"],
    )
