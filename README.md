# Wildfire Activity and Probability Dashboard

This repository contains a Streamlit dashboard for analyzing wildfire activity across the western United States. The project combines satellite active-fire detections, historical fire records, environmental data, and an experimental fire probability model.

The current dashboard starts with recent NASA FIRMS VIIRS active-fire detections. The next major feature is a county-month fire probability model that estimates whether at least one wildfire is likely to occur in a county during a selected month.

> Research and visualization use only. This project is not an official fire forecast, evacuation tool, or emergency warning system.

## Planned dashboard pages

### 1. Live Fire Activity

Shows recent satellite-detected active-fire pixels across the western United States. The page supports filters for fire radiative power, confidence, and map display style.

Current data source:

- NASA FIRMS VIIRS active-fire detections
- Recent near-real-time snapshot saved as Parquet

Important limitation:

- A FIRMS point is a satellite-detected thermal anomaly pixel, not an official fire perimeter or burned-area estimate.

### 2. Fire Probability Model

The first modeling page will estimate monthly county-level fire occurrence probability.

Initial prediction target:

```text
Probability that at least one wildfire occurs in a county during a given month.
```

Initial spatial and temporal unit:

```text
Spatial unit: county
Time unit: month
Region: western United States
```

Planned models:

- Logistic regression baseline
- Random forest or XGBoost model
- Probability calibration
- Time-based validation

Planned evaluation metrics:

- ROC-AUC
- PR-AUC
- Brier score
- Calibration curve
- Confusion matrix at selected threshold

### 3. Historical Event Explorer

Planned page for exploring major wildfire events such as the Camp Fire, Dixie Fire, August Complex, Bootleg Fire, Marshall Fire, and Smokehouse Creek Fire.

Planned outputs:

- Event map
- Daily detection-count curve
- Daily total fire radiative power curve
- Event-level summary metrics

### 4. Fire Trends

Planned page for long-term summaries of satellite-observed fire activity.

Planned outputs:

- Annual detection counts
- Annual total fire radiative power
- Seasonal timing summaries
- State-level comparisons

Sensor sources should be kept separate in trend plots because adding new satellites can change detection counts independently of real changes in fire activity.

### 5. Fire Hazard Context

Planned page for environmental layers related to fire behavior.

Candidate layers:

- Drought class
- Temperature anomaly
- Precipitation anomaly
- Vapor pressure deficit
- Wind
- Fuel or vegetation type
- Elevation, slope, and aspect
- Current fire perimeters

### 6. Data and Methods

Planned page for documenting data sources, processing steps, model design, validation, limitations, and citations.

## Current repository structure

```text
fire-hazard-dashboard/
  app.py
  fetch_firms.py
  requirements.txt
  pages/
    1_Live_Fires.py
  data/
    fires/
      viirs_west_recent.parquet
      viirs_west_<year>.parquet
```

## Target repository structure

```text
fire-hazard-dashboard/
  app.py
  pages/
    1_Live_Fire_Activity.py
    2_Fire_Probability.py
    3_Historical_Event_Explorer.py
    4_Fire_Trends.py
    5_Fire_Hazard_Context.py
    6_Data_and_Methods.py

  scripts/
    fetch_firms.py
    build_fire_labels.py
    build_fire_features.py
    train_fire_probability_model.py
    evaluate_fire_probability_model.py

  data/
    active_fire/
    labels/
    features/
    model_outputs/
    perimeters/
    context/
    derived/

  models/
  assets/
  requirements.txt
  README.md
  LICENSE
```

## Local setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Fetch recent FIRMS data

NASA FIRMS requires a map key.

Set the environment variable:

```bash
export FIRMS_MAP_KEY="your-map-key"
```

On Windows command prompt:

```cmd
set FIRMS_MAP_KEY=your-map-key
```

Fetch recent detections:

```bash
python fetch_firms.py --recent
```

Fetch one historical year:

```bash
python fetch_firms.py --year 2020
```

Fetch the full archive supported by the script:

```bash
python fetch_firms.py --all
```

## Near-term to-do list

- [ ] Polish the homepage.
- [ ] Add this README.
- [ ] Reorganize data folders.
- [ ] Keep more FIRMS columns, including sensor/source and processing fields.
- [ ] Add state, date, and day/night filters to the live fire page.
- [ ] Add a legend and top-FRP detection table.
- [ ] Add automated recent-data refresh with GitHub Actions.
- [ ] Build county-month fire occurrence labels.
- [ ] Build county-month environmental features.
- [ ] Train the first fire probability model.
- [ ] Add the Fire Probability Streamlit page.

## Limitations

FIRMS detections are satellite-detected thermal anomaly pixels. They are not official fire perimeters, burned-area estimates, or structure-level risk estimates. Fire radiative power is not the same as fire size. Satellite observations can be affected by overpass timing, cloud cover, smoke, sensor differences, and processing changes.

The planned probability model will estimate relative monthly fire occurrence probability from historical data and environmental features. It should not be used for emergency decisions.

## License

Add a license before public release.