import streamlit as st


st.set_page_config(
    page_title="Wildfire Hazard Dashboard",
    layout="wide",
)


st.title("Wildfire Hazard Dashboard — US West")

st.write(
    "This dashboard maps wildfire activity across the western United States using "
    "satellite fire detections from NASA. It shows both current fire activity and "
    "the major fire events of the past decade, rendered as interactive maps."
)

st.divider()

st.header("Sections")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Live Fires")
    st.write(
        "The most recent VIIRS satellite fire detections across the US West, shown "
        "as glowing points or a heatmap on a dark map. Points are colored by how "
        "recently the fire was detected and sized by fire intensity."
    )
    st.caption("Status: available")

with col_b:
    st.subheader("Historical Event Explorer")
    st.write(
        "Famous wildfires since 2012 — the Camp Fire, Dixie, August Complex, and "
        "more — shown over the true-color satellite image of the day, so the smoke "
        "plumes are visible beneath the detections."
    )
    st.caption("Status: in progress")

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Fire Trends")
    st.write(
        "How fire activity has changed since 2012: annual detection counts, total "
        "fire radiative power, and the shifting timing of the fire season."
    )
    st.caption("Status: planned")

with col_d:
    st.subheader("About the data")
    st.write(
        "Fire detections come from the NASA FIRMS VIIRS 375 m active fire product. "
        "Each detection marks a satellite-sensed pixel of active burning, not an "
        "official fire perimeter. Satellite imagery is from NASA GIBS."
    )
    st.caption("Source: NASA FIRMS / GIBS")

st.divider()

st.write("Use the sidebar to open any available section.")
