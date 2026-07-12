import streamlit as st

st.set_page_config(page_title="AI Travel Concierge", page_icon="✈️")

st.title("✈️ AI Travel Concierge")
st.subheader("Team: ElevateX")

st.write("Welcome! This AI agent will help users plan personalized trips.")

destination = st.text_input("Destination")
budget = st.number_input("Budget (₹)", min_value=1000)
travel_style = st.selectbox(
    "Travel Style",
    ["Adventure", "Luxury", "Budget", "Family", "Solo"]
)

if st.button("Generate Itinerary"):
    st.success("✅ Week 1 Prototype")
    st.write(f"📍 Destination: {destination}")
    st.write(f"💰 Budget: ₹{budget}")
    st.write(f"🎒 Travel Style: {travel_style}")

    st.info("AI itinerary generation will be implemented in the next milestone.")