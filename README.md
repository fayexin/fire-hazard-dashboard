# Wildfire Activity Dashboard

A Streamlit dashboard for mapping recent NASA FIRMS VIIRS active-fire detections across the western United States.

Current version:
- Live Fire Activity map
- FIRMS detection filters
- FRP and detection-count summaries
- Top high-FRP detection table

Not included:
- Official fire perimeters
- Evacuation guidance
- Fire probability model
- Fire-size estimates

Data source:
- NASA FIRMS VIIRS active-fire detections

Local setup:
pip install -r requirements.txt
set FIRMS_MAP_KEY=your_key
python fetch_firms.py --recent --source VIIRS_SNPP_NRT
streamlit run app.py