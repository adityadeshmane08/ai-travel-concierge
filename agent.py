import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

from tools import create_pdf_tool, create_web_tool


def create_agent(retriever):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    pdf_tool = create_pdf_tool(retriever)
    web_tool = create_web_tool(st.secrets["TAVILY_API_KEY"])

    agent = initialize_agent(
        tools=[pdf_tool, web_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
