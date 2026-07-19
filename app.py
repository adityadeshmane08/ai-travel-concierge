import streamlit as st
import os
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})



llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key
)
question = st.text_input(
    "Ask anything about travelling in India"
)

if question:

    docs = retriever.invoke(question)

pdf_context = "\n\n".join(
    doc.page_content for doc in docs
)

web_results = tavily.search(
    query=question,
    max_results=3
)

web_context = "\n\n".join(
    [r["content"] for r in web_results["results"]]
)


prompt = f"""
You are an AI Travel Concierge.

Use BOTH sources.

PDF Information:
{pdf_context}

Latest Web Information:
{web_context}

Question:
{question}

If the answer is available in the PDF, prefer it.
Otherwise use the web information.
"""

response = llm.invoke(prompt)
st.write(response.content)

import streamlit as st
from datetime import date

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Travel Concierge",
    page_icon="✈️",
    layout="wide"
)



# ----------------------------
# Header
# ----------------------------
st.title("✈️ AI Travel Concierge")
st.markdown(
    """
Welcome to **ElevateX's AI Travel Concierge**.

Plan your dream trip effortlessly by entering your travel details.

⚡ *AI itinerary generation will be available in the next milestone.*
"""
)

st.divider()

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    destination = st.text_input("📍 Destination")

    start_date = st.date_input(
        "📅 Start Date",
        value=date.today()
    )

    end_date = st.date_input(
        "📅 End Date",
        value=date.today()
    )

    travelers = st.number_input(
        "👨‍👩‍👧 Number of Travelers",
        min_value=1,
        max_value=20,
        value=2
    )

with col2:
    budget = st.slider(
        "💰 Budget (₹)",
        5000,
        200000,
        30000,
        step=1000
    )

    travel_style = st.selectbox(
        "🎒 Travel Style",
        [
            "Adventure",
            "Luxury",
            "Budget",
            "Family",
            "Solo",
            "Business"
        ]
    )

    interests = st.multiselect(
        "❤️ Interests",
        [
            "Beaches",
            "Mountains",
            "Food",
            "Shopping",
            "Historical Places",
            "Nightlife",
            "Wildlife",
            "Photography"
        ]
    )

st.divider()

if st.button("✨ Generate Travel Plan", use_container_width=True):

    st.success("Week 1 Prototype Successfully Running")

    st.markdown("## 🧳 Trip Summary")

    st.write(f"**Destination:** {destination}")
    st.write(f"**Travel Dates:** {start_date} → {end_date}")
    st.write(f"**Travelers:** {travelers}")
    st.write(f"**Budget:** ₹{budget}")
    st.write(f"**Travel Style:** {travel_style}")

    if interests:
        st.write("**Interests:**", ", ".join(interests))

    st.info(
        "🤖 AI itinerary generation, hotel recommendations, weather updates, and route optimization will be implemented in Week 2."
    )

st.divider()

st.markdown(
    """
### 🌟 Upcoming Features

- 🤖 AI-generated travel itinerary
- 🏨 Hotel recommendations
- 🍽️ Restaurant suggestions
- 🌤️ Weather forecasting
- 💸 Budget optimization
- 🗺️ Maps & route planning
- 🎟️ Tourist attraction recommendations
- 💬 Conversational AI Travel Assistant

---
Made with ❤️ by **Team ElevateX**
"""
)
