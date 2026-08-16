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
    system_prefix = (
        f"You are a helpful AI travel assistant. Today's date is {today}. "
        "When a user gives a relative date like 'this month', 'next month', "
        "or 'next week', convert it to an explicit YYYY-MM or YYYY-MM-DD "
        "value yourself before calling any tool - never pass a relative "
        "phrase like 'this month' directly as a tool argument."
    )

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
        max_iterations=6,
        early_stopping_method="generate",
    )

    return agent
