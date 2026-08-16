import streamlit as st
from datetime import date, datetime
from dotenv import load_dotenv
import re

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from agent import create_agent
from tools import search_flight_data, format_flight_results
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


# ----------------------------
# Theme
# ----------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-void: #0a0e17;
    --bg-panel: rgba(19, 27, 46, 0.55);
    --bg-panel-solid: #131b2e;
    --border-glow: rgba(34, 211, 238, 0.28);
    --accent-cyan: #22d3ee;
    --accent-violet: #a78bfa;
    --accent-amber: #fbbf24;
    --text-primary: #e7ecf7;
    --text-muted: #8993ac;
    --success: #34d399;
}

html, body, .stApp {
    background:
        radial-gradient(
            ellipse 80% 50% at 20% -10%,
            rgba(34,211,238,0.10),
            transparent
        ),
        radial-gradient(
            ellipse 60% 40% at 90% 10%,
            rgba(167,139,250,0.10),
            transparent
        ),
        var(--bg-void) !important;

    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

@media (prefers-reduced-motion: no-preference) {
    .stApp {
        animation: bgshift 30s ease-in-out infinite alternate;
    }

    @keyframes bgshift {
        0% {
            background-position: 0% 0%, 100% 0%, 0 0;
        }

        100% {
            background-position: 5% 5%, 95% -5%, 0 0;
        }
    }
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    display: block;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.1;
    background: linear-gradient(
        90deg,
        #ffffff 0%,
        var(--accent-cyan) 55%,
        var(--accent-violet) 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: var(--text-muted);
    font-size: 1.02rem;
    max-width: 640px;
}

.hud-line {
    height: 1px;
    margin: 1.4rem 0 1.6rem 0;
    background: linear-gradient(
        90deg,
        var(--accent-cyan),
        transparent 70%
    );
    opacity: 0.5;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        var(--accent-cyan),
        var(--accent-violet)
    );

    color: #08101f;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;

    box-shadow: 0 0 18px rgba(34, 211, 238, 0.35);

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 26px rgba(167, 139, 250, 0.5);
    color: #08101f;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stChatInput"] textarea,
textarea {
    background: var(--bg-panel-solid) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: var(--bg-panel-solid) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
}

[data-baseweb="tag"] {
    background: rgba(34, 211, 238, 0.18) !important;
    border: 1px solid var(--accent-cyan) !important;
}

[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent-cyan) !important;
    box-shadow: 0 0 10px var(--accent-cyan);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0d1424,
        #0a0e17
    ) !important;

    border-right: 1px solid var(--border-glow);
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(6px);
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 10px !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-glow);
    border-radius: 12px;
    backdrop-filter: blur(6px);
}

/* Glass panel */
.glass-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 1.4rem 1.5rem 0.6rem 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}

/* Boarding-pass itinerary card */
.boarding-pass {
    position: relative;
    background: linear-gradient(
        160deg,
        rgba(19,27,46,0.9),
        rgba(10,14,23,0.9)
    );

    border: 1px solid var(--accent-cyan);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0 1.4rem 0;

    box-shadow:
        0 0 30px rgba(34, 211, 238, 0.18),
        inset 0 0 40px rgba(167,139,250,0.05);

    overflow: hidden;
}

.boarding-pass::before {
    content: "";
    position: absolute;

    top: -10px;
    left: 38px;

    width: 20px;
    height: 20px;

    background: var(--bg-void);
    border-radius: 50%;

    box-shadow: 0 0 0 1px var(--accent-cyan);
}

.boarding-pass::after {
    content: "";

    position: absolute;

    bottom: -10px;
    left: 38px;

    width: 20px;
    height: 20px;

    background: var(--bg-void);
    border-radius: 50%;

    box-shadow: 0 0 0 1px var(--accent-cyan);
}

.bp-topline {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}

.bp-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--accent-amber);
}

.bp-route {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-cyan);
}

.bp-divider {
    border-top: 1px dashed rgba(139, 148, 172, 0.4);
    margin: 0.6rem 0 1rem 0;
}

.bp-content {
    color: var(--text-primary);
    line-height: 1.55;
}

.bp-content strong {
    color: var(--accent-cyan);
}

.log-item {
    border-left: 2px solid var(--accent-cyan);
    padding: 0.35rem 0 0.35rem 0.7rem;
    margin-bottom: 0.4rem;
}

.log-item .log-route {
    color: var(--accent-cyan);
    font-weight: 600;
}

.log-item .log-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
}

.lang-badge {
    display: inline-block;

    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;

    color: var(--accent-violet);

    border: 1px solid var(--accent-violet);
    border-radius: 999px;

    padding: 0.15rem 0.7rem;
    margin-bottom: 0.6rem;
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


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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

    from the current message plus previous conversation.
    """

    combined_parts = []

    for message in history:
        if isinstance(message, dict) and message.get("content"):
            combined_parts.append(message["content"])

    combined_parts.append(text)

    combined = "\n".join(combined_parts)

    # --------------------------------------------------------
    # Route
    #
    # Examples:
    #   Pune to Mumbai
    #   from Pune to Mumbai
    #   flight from Pune to Mumbai
    # --------------------------------------------------------

    route_match = re.search(
        r"(?:from\s+)?"
        r"([A-Za-z][A-Za-z .'-]{1,40}?)"
        r"\s+to\s+"
        r"([A-Za-z][A-Za-z .'-]{1,40}?)"
        r"(?=\s+(?:in|on|for|this|next|at)\b|[,.!?]|$)",
        combined,
        re.IGNORECASE,
    )

    origin = None
    destination = None

    if route_match:
        origin = route_match.group(1).strip()
        destination = route_match.group(2).strip()

    # --------------------------------------------------------
    # Exact date
    #
    # Examples:
    #   6 September 2026
    #   6th September 2026
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
    #
    # Examples:
    #   06/09/2026
    #   06-09-2026
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
    #
    # Example:
    #   September 2026
    #
    # IMPORTANT:
    # If user only says "September", we also infer 2026
    # because the current application flow is for 2026.
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
    #
    # Example:
    #   September
    #
    # Use 2026 as the default year for this app.
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

        answer = f"Something went wrong: {exc}"

        response = {
            "output": answer
        }

    return answer, response


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<span class="eyebrow">TEAM ELEVATEX // MISSION CONTROL</span>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">AI Travel Concierge</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-sub">'
    'Plan trips across India with an agent that pulls live web '
    'context, cached flight prices, and a curated travel guide - '
    'and talks back in whatever language you use.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hud-line"></div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<span class="eyebrow">🛰️ Flight Log</span>',
        unsafe_allow_html=True,
    )

    saved = db.get_saved_searches()

    if not saved:

        st.caption(
            "No saved trips yet. Generate a plan to log it here."
        )

    for row in saved:

        st.markdown(
            f"""
            <div class="log-item">
                <div class="log-route">
                    {row['origin']} → {row['destination']}
                </div>

                <div class="log-meta">
                    {row['start_date']} ·
                    {row['travelers']} pax ·
                    ₹{row['budget']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View manifest"):

            st.write(row["itinerary"])

            if st.button(
                "Delete entry",
                key=f"del_{row['id']}",
            ):

                db.delete_search(row["id"])

                st.rerun()


# ============================================================
# TRIP PLANNING FORM
# ============================================================

st.markdown(
    '<div class="glass-panel">',
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="eyebrow">Mission Parameters</span>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)


with col1:

    origin = st.text_input(
        "🛫 Departure City",
        value="Mumbai",
    )

    destination = st.text_input(
        "📍 Destination",
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

    else:

        flight_error = None
        flight_offers = []

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

Use the PDF Travel Guide tool for background on {destination} in India.

Use the Web Search tool for anything current such as weather,
events, or prices that is not covered by the PDF.

Then produce a day-by-day itinerary that fits the budget and
interests.

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
        # Boarding pass
        # ----------------------------------------------------

        boarding_pass_html = (
            '<div class="boarding-pass">'

            '<div class="bp-topline">'

            '<span class="bp-eyebrow">'
            'TRIP MANIFEST'
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

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        with st.expander(
            "🔍 Debug: raw agent response"
        ):

            st.json(response)

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
    '<span class="eyebrow">Live Channel</span>',
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
        "🧑‍🚀"
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
    "Ask anything - kisi bhi bhasha mein poochh sakte hain..."
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
        avatar="🧑‍🚀",
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

        with st.expander(
            "🔍 Debug: raw agent response"
        ):

            st.json(response)

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
<span class="eyebrow">Upcoming Systems</span>

- 🏨 Hotel recommendations
- 🍽️ Restaurant suggestions
- 🌤️ Weather forecasting
- 🗺️ Maps & route planning

---

Made with ❤️ by **Team ElevateX**
""",
    unsafe_allow_html=True,
)
