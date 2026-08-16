# AI Travel Concierge

An AI-powered travel assistant for planning trips in India. Combines a
RAG chatbot over an India travel guide, live web search, and cached
flight price data into a single LangChain agent, wrapped in a Streamlit UI.

## Features

- 🤖 Agent with 3 tools: PDF Travel Guide (RAG), Web Search (Tavily),
  Flight Search (Travelpayouts)
- 🧳 Trip-planning form that generates a real day-by-day itinerary
- 💾 SQLite storage of past searches, browsable in the sidebar
- 💬 Free-form chat for ad-hoc travel questions

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create `.streamlit/secrets.toml` with:
   ```toml
   GROQ_API_KEY = "your-groq-key"
   TAVILY_API_KEY = "your-tavily-key"
   TRAVELPAYOUTS_TOKEN = "your-travelpayouts-token"
   ```

   - Groq: https://console.groq.com
   - Tavily: https://tavily.com
   - Travelpayouts (free, self-serve, no traffic minimum for this API):
     1. Go to https://www.travelpayouts.com and click Sign Up (free).
     2. Verify your email and log in.
     3. Go to your Dashboard -> Tools -> API (or directly
        https://www.travelpayouts.com/programs/100/tools/api).
     4. Copy your API token and paste it into secrets.toml as
        TRAVELPAYOUTS_TOKEN.

   > Note: Amadeus's self-service developer portal was shut down on
   > July 17, 2026, so this project uses Travelpayouts instead. If you'd
   > rather use a different provider, just swap out
   > travelpayouts_client.py for a client of your choice.

3. Run locally:
   ```
   streamlit run app.py
   ```

## Project structure

- `app.py` — Streamlit UI: trip form, saved-trips sidebar, chat box
- `agent.py` — builds the LangChain structured-chat agent with all 3 tools
- `tools.py` — tool definitions (PDF retrieval, web search, flight search)
- `travelpayouts_client.py` — Travelpayouts city lookup + cached flight
  price search
- `db.py` — SQLite persistence for saved trip searches
- `Travel-guide-for-India.pdf` — source document for the RAG tool

## Flight-search architecture

- `tools.py` contains the single shared `search_flight_data()` function used by both the **Generate Travel Plan** button and the chat agent's **Flight Search** tool.
- The Generate Travel Plan flow runs that shared search once, shows the returned cached offers, and passes the exact results into the itinerary prompt so the itinerary cannot invent different flight options.
- Chat flight requests call the same shared function through the Flight Search tool.
- Chat follow-up context no longer duplicates the current user message before it reaches the agent.

## Notes

- Travelpayouts flight data is served from a cache of recent user
  searches, not a live shopping call - fine for a demo, but don't
  present it as a guaranteed bookable price.
- The Chroma vector index is built once and cached via
  `@st.cache_resource`, so it won't rebuild on every interaction.
