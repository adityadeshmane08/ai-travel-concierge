import streamlit as st
from datetime import date, datetime
from dotenv import load_dotenv
import re

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from agent import create_agent, create_fallback_llm
from tools import search_flight_data, format_flight_results
import weather_client
import db


load_dotenv()
db.init_db()


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Travel Concierge",
    page_icon="🛰️",
    layout="wide",
)


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-page: #f5f7fb;
    --bg-card: #ffffff;
    --border-soft: #e6e9f2;
    --border-hover: #c7d0e8;
    --primary: #0d6efd;
    --primary-dark: #0b57d0;
    --accent: #ff6b35;
    --accent-dark: #e5572a;
    --success: #1aa260;
    --text-primary: #101828;
    --text-muted: #667085;
    --shadow-card: 0 2px 10px rgba(16, 24, 40, 0.06);
    --shadow-card-hover: 0 8px 24px rgba(16, 24, 40, 0.10);
}

html, body, .stApp {
    background: var(--bg-page) !important;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: -0.01em;
    color: var(--text-primary);
}

.eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--primary);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.15;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.hero-title .accent-word {
    color: var(--primary);
}

.hero-sub {
    color: var(--text-muted);
    font-size: 1.08rem;
    max-width: 680px;
    line-height: 1.65;
    margin-bottom: 0.6rem;
}

.hud-line {
    height: 1px;
    margin: 1.8rem 0;
    background: var(--border-soft);
}

/* Trust badge row */
.trust-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 1rem 0 0.4rem 0;
}

.trust-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #eef4ff;
    color: var(--primary-dark);
    border: 1px solid #d6e4ff;
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Feature / How-it-works grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1rem;
    margin: 0.8rem 0 1.6rem 0;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    box-shadow: var(--shadow-card);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.feature-card:hover {
    box-shadow: var(--shadow-card-hover);
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
    display: block;
}

.feature-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.3rem;
    color: var(--text-primary);
}

.feature-desc {
    color: var(--text-muted);
    font-size: 0.88rem;
    line-height: 1.5;
}

/* Sleek Buttons */
.stButton > button {
    background: var(--accent);
    color: #ffffff;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    box-shadow: 0 4px 14px rgba(255, 107, 53, 0.28);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: var(--accent-dark);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(255, 107, 53, 0.35);
    color: #ffffff;
}

[data-testid="stDownloadButton"] > button {
    background: #ffffff;
    color: var(--primary);
    border: 1.5px solid var(--primary);
    font-weight: 600;
    border-radius: 10px;
    box-shadow: none;
}

[data-testid="stDownloadButton"] > button:hover {
    background: #eef4ff;
    color: var(--primary-dark);
    transform: none;
}

/* Inputs & Form Elements */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stChatInput"] textarea,
textarea {
    background: #ffffff !important;
    color: var(--text-primary) !important;
    border: 1.5px solid var(--border-soft) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.12);
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid var(--border-soft) !important;
    border-radius: 10px !important;
}

[data-baseweb="tag"] {
    background: #eef4ff !important;
    border: 1px solid var(--primary) !important;
    color: var(--primary-dark) !important;
    border-radius: 6px !important;
}

/* Booking-style search panel */
.glass-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.6rem 1.8rem 1.2rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-card);
}

/* Itinerary result card */
.boarding-pass {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-left: 5px solid var(--primary);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0 1.6rem 0;
    box-shadow: var(--shadow-card-hover);
    overflow: hidden;
}

.bp-topline {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.9rem;
}

.bp-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--accent);
    text-transform: uppercase;
}

.bp-route {
    font-family: 'Poppins', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary);
}

.bp-divider {
    border-top: 1px dashed var(--border-soft);
    margin: 0.8rem 0 1.2rem 0;
}

.bp-content {
    color: var(--text-primary);
    line-height: 1.7;
}

.bp-content strong {
    color: var(--primary-dark);
}

/* Weather card */
.weather-card {
    background: linear-gradient(135deg, #eaf2ff 0%, #f5f9ff 100%);
    border: 1px solid #d6e4ff;
    border-radius: 16px;
    padding: 1.3rem 1.6rem;
    margin: 0 0 1.4rem 0;
}

.weather-current {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.9rem;
}

.weather-current .icon {
    font-size: 2.4rem;
}

.weather-current .temp {
    font-family: 'Poppins', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
}

.weather-current .desc {
    color: var(--text-muted);
    font-size: 0.95rem;
}

.weather-days {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
}

.weather-day {
    background: #ffffff;
    border: 1px solid #e2ecff;
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    text-align: center;
    min-width: 78px;
    font-size: 0.82rem;
}

.weather-day .d-icon {
    font-size: 1.2rem;
    display: block;
}

.weather-day .d-temp {
    font-weight: 700;
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border-soft);
}

.trip-log-card {
    background: #f8faff;
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
}

.trip-log-card:hover {
    border-color: var(--primary);
    background: #eef4ff;
}

.lang-badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--primary-dark);
    border: 1px solid #d6e4ff;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    margin-bottom: 0.9rem;
    background: #eef4ff;
}

.site-footer {
    color: var(--text-muted);
    font-size: 0.85rem;
    text-align: center;
    padding: 1rem 0 0.4rem 0;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============================================================
# RAG + AGENT
# ============================================================

@st.cache_resource(show_spinner="Calibrating knowledge base...")
def get_agent():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    loader = PyPDFLoader("Travel-guide-for-India.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = text_splitter.split_documents(documents)

    db_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    retriever = db_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return create_agent(retriever)


agent = get_agent()


@st.cache_resource
def get_fallback_llm():
    return create_fallback_llm()


fallback_llm = get_fallback_llm()


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# FLIGHT REQUEST EXTRACTION
# ============================================================

# ============================================================
# FLIGHT REQUEST EXTRACTION
# ============================================================

# ============================================================
# FLIGHT REQUEST EXTRACTION
# ============================================================

def extract_flight_request(
    text: str,
    history: list[dict],
):
    """
    Extract:
    origin
    destination
    departure date/month
    from the current message plus previous conversation (newest first).
    """

    # 1. Put the newest message FIRST
    combined_parts = [text]

    # 2. Add history in REVERSE order, only looking at what the user typed
    for message in reversed(history):
        if isinstance(message, dict) and message.get("content") and message.get("role") == "user":
            combined_parts.append(message["content"])

    combined = "\n".join(combined_parts)
    
    # Remove keywords so the regex doesn't swallow them
    clean_combined = re.sub(
        r"(?i)\b(flight(?:s)?|price|cost|fare|ticket(?:s)?)\b", 
        "", 
        combined
    ).strip()

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    route_match = re.search(
        r"(?:from\s+)?"
        r"([A-Za-z][A-Za-z .'-]{1,25}?)"
        r"\s+to\s+"
        r"([A-Za-z][A-Za-z .'-]{1,25}?)"
        r"(?=\s+(?:in|on|for|this|next|at)\b|[,.!?]|$)",
        clean_combined,
        re.IGNORECASE,
    )

    origin = None
    destination = None

    if route_match:
        origin = route_match.group(1).strip()
        destination = route_match.group(2).strip()

    # --------------------------------------------------------
    # Exact date
    # --------------------------------------------------------

    date_match = re.search(
        r"\b"
        r"(\d{1,2})"
        r"(?:st|nd|rd|th)?"
        r"\s+"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+"
        r"(\d{4})"
        r"\b",
        combined,
        re.IGNORECASE,
    )

    departure_date = None

    if date_match:
        day, month, year = date_match.groups()

        try:
            dt = datetime.strptime(
                f"{day} {month} {year}",
                "%d %B %Y",
            )

            departure_date = dt.strftime("%Y-%m-%d")

        except ValueError:
            departure_date = None

    # --------------------------------------------------------
    # Numeric date
    # --------------------------------------------------------

    if not departure_date:

        numeric_match = re.search(
            r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
            combined,
        )

        if numeric_match:

            day, month, year = numeric_match.groups()

            try:
                dt = datetime(
                    int(year),
                    int(month),
                    int(day),
                )

                departure_date = dt.strftime("%Y-%m-%d")

            except ValueError:
                departure_date = None

    # --------------------------------------------------------
    # Month + year
    # --------------------------------------------------------

    if not departure_date:

        month_year_match = re.search(
            r"\b"
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+"
            r"(\d{4})"
            r"\b",
            combined,
            re.IGNORECASE,
        )

        if month_year_match:

            month, year = month_year_match.groups()

            try:
                dt = datetime.strptime(
                    f"1 {month} {year}",
                    "%d %B %Y",
                )

                departure_date = dt.strftime("%Y-%m")

            except ValueError:
                departure_date = None

    # --------------------------------------------------------
    # Month only
    # --------------------------------------------------------

    if not departure_date:

        month_only_match = re.search(
            r"\b"
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\b",
            combined,
            re.IGNORECASE,
        )

        if month_only_match:

            month = month_only_match.group(1)

            try:
                dt = datetime.strptime(
                    f"1 {month} 2026",
                    "%d %B %Y",
                )

                departure_date = dt.strftime("%Y-%m")

            except ValueError:
                departure_date = None

    return origin, destination, departure_date


# ============================================================
# RUN AGENT / FLIGHT CHAT
# ============================================================

def run_agent(
    prompt_text: str,
    memory_context: bool = True,
) -> tuple[str, dict]:

    """
    Flight questions are handled directly using the SAME shared
    search_flight_data() function used by Generate Travel Plan.

    Other questions are sent to the normal AI agent.
    """

    # --------------------------------------------------------
    # Previous conversation
    # --------------------------------------------------------

    history = st.session_state.chat_history[:-1]

    # --------------------------------------------------------
    # Build normal agent prompt
    # --------------------------------------------------------

    if memory_context and history:

        recent = history[-6:]

        context_block = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent
        )

        full_prompt = (
            f"Conversation so far:\n"
            f"{context_block}\n\n"
            f"User: {prompt_text}"
        )

    else:
        full_prompt = prompt_text

    # --------------------------------------------------------
    # Detect flight request
    # --------------------------------------------------------

    flight_keywords = [
        "flight",
        "flights",
        "airfare",
        "air fare",
        "flight cost",
        "flight price",
        "ticket price",
        "air ticket",
    ]

    current_lower = prompt_text.lower()

    previous_text = "\n".join(
        message["content"]
        for message in history
        if isinstance(message, dict)
        and message.get("content")
    )

    previous_lower = previous_text.lower()

    explicit_flight_request = any(
        keyword in current_lower
        for keyword in flight_keywords
    )

    month_or_date_followup = bool(
        re.search(
            r"\b(?:january|february|march|april|may|june|july|"
            r"august|september|october|november|december)\b",
            current_lower,
        )
        or re.search(
            r"\b\d{1,2}(?:st|nd|rd|th)?\b",
            current_lower,
        )
        or re.search(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
            current_lower,
        )
    )

    previous_flight_request = any(
        keyword in previous_lower
        for keyword in flight_keywords
    )

    is_flight_request = (
        explicit_flight_request
        or (
            previous_flight_request
            and month_or_date_followup
        )
    )

    # ========================================================
    # DIRECT FLIGHT SEARCH
    # ========================================================

    if is_flight_request:

        origin, destination, departure_date = extract_flight_request(
            prompt_text,
            history,
        )

        # ----------------------------------------------------
        # Missing route
        # ----------------------------------------------------

        if not origin or not destination:

            answer = (
                "Sure! What are your departure and destination cities? "
                "For example: Pune to Mumbai."
            )

            return answer, {
                "type": "flight_clarification",
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
            }

        # ----------------------------------------------------
        # Missing date
        # ----------------------------------------------------

        if not departure_date:

            answer = (
                f"Sure! What date or month would you like to fly "
                f"from {origin} to {destination}?"
            )

            return answer, {
                "type": "flight_clarification",
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
            }

        # ----------------------------------------------------
        # SAME SHARED FLIGHT SEARCH
        # ----------------------------------------------------

        try:

            offers = search_flight_data(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                token=st.secrets["TRAVELPAYOUTS_TOKEN"],
            )

            # ------------------------------------------------
            # Format using SAME shared formatter
            # ------------------------------------------------

            if offers:

                answer = format_flight_results(offers)

            else:

                answer = (
                    f"I couldn't find cached flight price data for "
                    f"{origin} → {destination} on {departure_date}.\n\n"
                    "The flight database uses cached prices, so this "
                    "does not necessarily mean flights are unavailable."
                )

            return answer, {
                "type": "flight_search",
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "offers": offers,
            }

        except Exception as exc:

            answer = (
                f"I couldn't search flights from {origin} to "
                f"{destination} right now.\n\n"
                f"Error: {exc}"
            )

            return answer, {
                "type": "flight_error",
                "error": str(exc),
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
            }

    # ========================================================
    # NORMAL AI AGENT
    # ========================================================

    try:

        response = agent.invoke(full_prompt)

        answer = (
            response.get("output") or ""
        ).strip()

        if not answer:

            steps = response.get(
                "intermediate_steps"
            ) or []

            if steps:

                last_action, last_observation = steps[-1]

                answer = (
                    f"(Ran out of reasoning steps, but here's what "
                    f"I found from **{last_action.tool}**:)\n\n"
                    f"{last_observation}"
                )

            else:

                answer = (
                    "I couldn't work out a full answer that time. "
                    "Could you rephrase your question?"
                )

    except Exception as exc:

        # The main agent enforces a strict JSON action format, which
        # Groq's hosted model occasionally fails to produce cleanly for
        # simple conversational questions (e.g. "best place in Pune?").
        # Rather than showing the user a raw API error, fall back to a
        # plain LLM call with no format constraints - it almost always
        # answers the actual travel question correctly.
        try:
            fallback_prompt = (
                "You are a friendly, knowledgeable AI travel assistant "
                "for India. Answer the user's question directly and "
                "conversationally, in the same language and script they "
                "used. If they're asking for a recommendation (best "
                "place to visit, what to see, etc.), give a specific, "
                "confident suggestion with a one-line reason - don't "
                "hedge or ask for tools. Never mention errors, JSON, "
                "or that something went wrong.\n\n"
                f"User: {prompt_text}"
            )

            fallback_result = fallback_llm.invoke(fallback_prompt)
            answer = (fallback_result.content or "").strip()

            if not answer:
                raise ValueError("empty fallback response")

            response = {"output": answer, "fallback_used": True}

        except Exception:
            answer = (
                "I couldn't quite work that out. Could you ask again - "
                "for example, a destination, a flight route, or "
                "something about the weather where you're headed?"
            )
            response = {"output": answer}

    return answer, response


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<span class="eyebrow">AI-Powered Trip Planning</span>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Plan your next trip with your '
    '<span class="accent-word">AI Travel Concierge</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-sub">'
    'Tell us where you want to go and we\'ll pull live flight prices, '
    'real weather forecasts, and a curated India travel guide into one '
    'day-by-day itinerary - built by an agent that talks back in '
    'whatever language you write in.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="trust-row">
    <span class="trust-badge">✅ Live cached flight prices</span>
    <span class="trust-badge">🌦️ Real-time weather data</span>
    <span class="trust-badge">📚 Curated India travel guide</span>
    <span class="trust-badge">🔒 Your searches, saved privately</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hud-line"></div>',
    unsafe_allow_html=True,
)

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<span class="eyebrow">How It Works</span>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">📝</span>
        <div class="feature-title">1. Tell us your trip</div>
        <div class="feature-desc">Departure city, destination, dates,
        budget, and the kind of trip you're after.</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🤖</span>
        <div class="feature-title">2. AI does the research</div>
        <div class="feature-desc">Flight prices, live weather, and
        destination knowledge are pulled together automatically.</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🗓️</span>
        <div class="feature-title">3. Get a real itinerary</div>
        <div class="feature-desc">A day-by-day plan that fits your
        budget - save it, export it, or keep chatting to refine it.</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# SIDEBAR FLIGHT LOG (Clean & Readable)
# ============================================================

with st.sidebar:

    st.markdown(
        '<span class="eyebrow">🧳 My Trips</span>',
        unsafe_allow_html=True,
    )

    saved = db.get_saved_searches()

    if not saved:

        st.caption(
            "No saved trips yet. Generate a plan to save it here."
        )

    for row in saved:

        # Clean, human-readable card layout without raw HTML exposure
        with st.container():
            st.markdown(f"**{row['origin']} ➔ {row['destination']}**")

            # Display readable metadata
            st.caption(
                f"📅 {row['start_date']} | "
                f"👥 {row['travelers']} Travelers | "
                f"💰 ₹{row['budget']}"
            )

        with st.expander("View trip details"):
            st.write(row["itinerary"])

            st.download_button(
                "⬇️ Download itinerary",
                data=row["itinerary"],
                file_name=f"{row['origin']}_to_{row['destination']}_itinerary.txt",
                mime="text/plain",
                key=f"dl_{row['id']}",
                use_container_width=True,
            )

            if st.button(
                "Delete entry",
                key=f"del_{row['id']}",
            ):
                db.delete_search(row["id"])
                st.rerun()

        st.markdown("---")


# ============================================================
# TRIP PLANNING FORM
# ============================================================

st.markdown(
    '<div class="glass-panel">',
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="eyebrow">Plan Your Trip</span>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)


with col1:

    origin = st.text_input(
        "🛫 Departure City",
        placeholder="e.g. Mumbai",
    )

    destination = st.text_input(
        "📍 Destination",
        placeholder="e.g. Goa",
    )

    start_date = st.date_input(
        "📅 Start Date",
        value=date.today(),
    )

    end_date = st.date_input(
        "📅 End Date",
        value=date.today(),
    )

    travelers = st.number_input(
        "👨‍👩‍👧 Number of Travelers",
        min_value=1,
        max_value=20,
        value=2,
    )


with col2:

    budget = st.slider(
        "💰 Budget (₹)",
        5000,
        200000,
        30000,
        step=1000,
    )

    travel_style = st.selectbox(
        "🎒 Travel Style",
        [
            "Adventure",
            "Luxury",
            "Budget",
            "Family",
            "Solo",
            "Business",
        ],
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
            "Photography",
        ],
    )


generate_clicked = st.button(
    "✨ Generate Travel Plan",
    use_container_width=True,
)

st.markdown(
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# GENERATE TRAVEL PLAN
# ============================================================

if generate_clicked:

    if not destination:

        st.warning(
            "Please enter a destination first."
        )

    elif not origin:

        st.warning(
            "Please enter a departure city first."
        )

    elif end_date < start_date:

        st.warning(
            "End date can't be before the start date - "
            "please double-check your travel dates."
        )

    elif budget < 1000:

        st.warning(
            "That budget looks too low to plan a trip around - "
            "try increasing it a bit."
        )

    else:

        flight_error = None
        flight_offers = []

        # ----------------------------------------------------
        # Real weather forecast for the destination
        # ----------------------------------------------------

        try:
            weather_forecast = weather_client.get_forecast(destination)
        except Exception:
            weather_forecast = None

        weather_context = weather_client.format_forecast_text(
            weather_forecast
        )

        # ----------------------------------------------------
        # SAME SHARED FLIGHT SEARCH FUNCTION
        # ----------------------------------------------------

        try:

            flight_offers = search_flight_data(
                origin=origin,
                destination=destination,
                departure_date=start_date.isoformat(),
                token=st.secrets["TRAVELPAYOUTS_TOKEN"],
            )

            flight_context = format_flight_results(
                flight_offers
            )

        except Exception as exc:

            flight_error = str(exc)

            flight_context = flight_error

        # ----------------------------------------------------
        # Itinerary prompt
        # ----------------------------------------------------

        prompt = f"""
Plan a trip with the following details:

- Departure city: {origin}
- Destination: {destination}
- Travel dates: {start_date} to {end_date}
- Number of travelers: {travelers}
- Budget: ₹{budget}
- Travel style: {travel_style}
- Interests: {", ".join(interests) if interests else "general sightseeing"}

IMPORTANT FLIGHT DATA:

The application has ALREADY run the shared Flight Search logic
for this trip.

Use the exact flight results below in the itinerary.

Do NOT call the Flight Search tool again for this request.

Do NOT invent flight options.

If the result says there is no cached data, say so clearly.

{flight_context}

REAL WEATHER DATA for {destination} (already fetched - do not call the
Weather Forecast tool again for this request, do not invent conditions):

{weather_context}

Use the PDF Travel Guide tool for background on {destination} in India.

Use the Web Search tool for anything current such as events or prices
that is not covered by the PDF or the weather data above.

Then produce a day-by-day itinerary that fits the budget and
interests, and factor the weather into your recommendations (e.g.
suggest indoor alternatives on rainy days, pack accordingly).

Mention the flight options found above.

Respond in English unless the destination or interests suggest otherwise.
"""

        # ----------------------------------------------------
        # Generate itinerary
        # ----------------------------------------------------

        with st.spinner(
            "Plotting flight paths and building itinerary..."
        ):

            itinerary_text, response = run_agent(
                prompt,
                memory_context=False,
            )

        # ----------------------------------------------------
        # Flight result display
        # ----------------------------------------------------

        if flight_error:

            st.warning(
                f"Flight search unavailable: {flight_error}"
            )

        elif flight_offers:

            with st.expander(
                "✈️ Flight options used for this plan",
                expanded=True,
            ):

                st.markdown(
                    format_flight_results(
                        flight_offers
                    )
                )

        else:

            st.info(
                "No cached flight price data was found "
                "for this route/date."
            )

        # ----------------------------------------------------
        # Weather card
        # ----------------------------------------------------

        if weather_forecast:

            current = weather_forecast["current"]

            days_html = "".join(
                f'<div class="weather-day">'
                f'<span class="d-icon">{d["icon"]}</span>'
                f'{d["date"][5:]}<br>'
                f'<span class="d-temp">{round(d["high"])}°</span>'
                f'/{round(d["low"])}°'
                f'</div>'
                for d in weather_forecast["daily"]
            )

            st.markdown(
                f"""
<div class="weather-card">
    <div class="weather-current">
        <span class="icon">{current['icon']}</span>
        <div>
            <div class="temp">{round(current['temp'])}°C</div>
            <div class="desc">{current['desc']} in
            {weather_forecast['place']} · Humidity {current['humidity']}%
            · Wind {round(current['wind'])} km/h</div>
        </div>
    </div>
    <div class="weather-days">{days_html}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        else:

            st.info(
                f"Couldn't fetch a live weather forecast for "
                f"{destination} - the itinerary will skip weather "
                f"specifics."
            )

        # ----------------------------------------------------
        # Boarding pass
        # ----------------------------------------------------

        boarding_pass_html = (
            '<div class="boarding-pass">'

            '<div class="bp-topline">'

            '<span class="bp-eyebrow">'
            'YOUR ITINERARY'
            '</span>'

            f'<span class="bp-route">'
            f'{origin.upper()} → {destination.upper()}'
            f'</span>'

            '</div>'

            '<div class="bp-divider"></div>'

            '<div class="bp-content">'

            f'{itinerary_text}'

            '</div>'

            '</div>'
        )

        st.markdown(
            boarding_pass_html,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Download this itinerary",
            data=itinerary_text,
            file_name=f"{origin}_to_{destination}_itinerary.txt",
            mime="text/plain",
        )

        # ----------------------------------------------------
        # Save search
        # ----------------------------------------------------

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

        st.success(
            "Logged to Flight Log ✅"
        )


# ============================================================
# CHAT
# ============================================================

st.markdown(
    '<div class="hud-line"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="eyebrow">Ask Anything</span>',
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="lang-badge">'
    '🌐 हिंदी · English · Hinglish · +more'
    '</span>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Display previous messages
# ------------------------------------------------------------

for msg in st.session_state.chat_history:

    avatar = (
        "🧑"
        if msg["role"] == "user"
        else "🤖"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar,
    ):

        st.markdown(
            msg["content"]
        )


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

user_msg = st.chat_input(
    "Ask anything about travel in India — weather, flight rates, "
    "best places to visit... (kisi bhi bhasha mein poochh sakte hain)"
)


if user_msg:

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_msg,
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑",
    ):

        st.markdown(user_msg)

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer, response = run_agent(
                user_msg,
                memory_context=True,
            )

        st.markdown(answer)

        

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="hud-line"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🔒</span>
        <div class="feature-title">Your data stays yours</div>
        <div class="feature-desc">Saved trips are stored locally to
        this app - nothing is shared with third parties.</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">💬</span>
        <div class="feature-title">Multilingual by default</div>
        <div class="feature-desc">Chat in English, Hindi, or Hinglish -
        the assistant replies in kind.</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <div class="feature-title">Always improving</div>
        <div class="feature-desc">Hotel and restaurant recommendations
        are on the roadmap next.</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="site-footer">AI Travel Concierge · '
    'Flight data is cached, not live-booked - confirm prices before '
    'purchase.</div>',
    unsafe_allow_html=True,
)
