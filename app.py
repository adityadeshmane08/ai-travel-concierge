import streamlit as st
from datetime import date
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from agent import create_agent
import db

load_dotenv()
db.init_db()

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Travel Concierge",
    page_icon="✈️",
    layout="wide"
)


# ----------------------------
# Cached setup: build the RAG index + agent ONCE, not on every rerun
# ----------------------------
@st.cache_resource(show_spinner="Setting up the travel assistant...")
def get_agent():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    loader = PyPDFLoader("Travel-guide-for-India.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    db_store = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = db_store.as_retriever(search_kwargs={"k": 3})

    return create_agent(retriever)


agent = get_agent()

# ----------------------------
# Header
# ----------------------------
st.title("✈️ AI Travel Concierge")
st.markdown(
    """
Welcome to **ElevateX's AI Travel Concierge**.

Fill in your trip details below and the AI agent will build you an
itinerary using the India travel guide, live web search, and real
flight prices.
"""
)

st.divider()

# ----------------------------
# Sidebar: saved searches
# ----------------------------
with st.sidebar:
    st.header("📌 Saved Trips")
    saved = db.get_saved_searches()
    if not saved:
        st.caption("No saved trips yet. Generate a plan to save it here.")
    for row in saved:
        with st.expander(f"{row['destination']} ({row['start_date']})"):
            st.write(f"**From:** {row['origin']}")
            st.write(f"**Travelers:** {row['travelers']}")
            st.write(f"**Budget:** ₹{row['budget']}")
            st.write(f"**Style:** {row['travel_style']}")
            st.write(row["itinerary"])
            if st.button("Delete", key=f"del_{row['id']}"):
                db.delete_search(row["id"])
                st.rerun()

# ----------------------------
# Trip Planning Form
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    origin = st.text_input("🛫 Departure City", value="Mumbai")
    destination = st.text_input("📍 Destination")

    start_date = st.date_input("📅 Start Date", value=date.today())
    end_date = st.date_input("📅 End Date", value=date.today())

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
        ["Adventure", "Luxury", "Budget", "Family", "Solo", "Business"]
    )

    interests = st.multiselect(
        "❤️ Interests",
        [
            "Beaches", "Mountains", "Food", "Shopping",
            "Historical Places", "Nightlife", "Wildlife", "Photography"
        ]
    )

st.divider()

if st.button("✨ Generate Travel Plan", use_container_width=True):
    if not destination:
        st.warning("Please enter a destination first.")
    else:
        prompt = f"""
Plan a trip with the following details:
- Departure city: {origin}
- Destination: {destination}
- Travel dates: {start_date} to {end_date}
- Number of travelers: {travelers}
- Budget: ₹{budget}
- Travel style: {travel_style}
- Interests: {", ".join(interests) if interests else "general sightseeing"}

Use the Flight Search tool to find real flight options from {origin} to
{destination} departing on {start_date}. Use the PDF Travel Guide tool for
background on {destination} in India, and the Web Search tool for anything
current (weather, events, prices) not covered by the PDF. Then produce a
day-by-day itinerary that fits the budget and interests, and mention the
flight options you found.
"""
        with st.spinner("Building your itinerary..."):
            try:
                response = agent.invoke(prompt)
                itinerary_text = response["output"]
            except Exception as exc:
                itinerary_text = f"Something went wrong while generating the plan: {exc}"

        st.markdown("## 🧳 Your Itinerary")
        st.write(itinerary_text)

        db.save_search(
            destination=destination,
            origin=origin,
            start_date=start_date,
            end_date=end_date,
            travelers=travelers,
            budget=budget,
            travel_style=travel_style,
            interests=interests,
            itinerary=itinerary_text,
        )
        st.success("Saved to your trip history in the sidebar ✅")

st.divider()

# ----------------------------
# Free-form chat with the agent
# ----------------------------
st.markdown("### 💬 Ask the Travel Assistant Anything")
question = st.text_input("e.g. 'What's the best time to visit Goa?'")

if question:
    with st.spinner("Thinking..."):
        response = agent.invoke(question)
    st.write(response["output"])

st.divider()

st.markdown(
    """
### 🌟 Upcoming Features

- 🏨 Hotel recommendations
- 🍽️ Restaurant suggestions
- 🌤️ Weather forecasting
- 🗺️ Maps & route planning

---
Made with ❤️ by **Team ElevateX**
"""
)
