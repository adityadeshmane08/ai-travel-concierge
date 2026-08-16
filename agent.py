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
    )

    return agent
