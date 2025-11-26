# Multi-State Land Use Emissions Analysis

[![ci](https://github.com/Deshraj-Jogiya/Multi-State-Land-Use-Emissions-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Deshraj-Jogiya/Multi-State-Land-Use-Emissions-Analysis/actions/workflows/ci.yml) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

An ETL and analysis pipeline over daily land-cover data for 5 U.S. states (Alabama, California, Texas, Washington, Florida): it builds
a daily net CO2 balance per state and land type, flags sudden land-cover shocks such as wildfires or bursts of urban expansion, and
tests whether Random Forest and Linear Regression can forecast the balance. The emission factors and a year-ahead forecast use the
EPA's real state greenhouse-gas inventory for land use (1990-2022); the daily land-cover files are simulated.

## Highlights

- **ETL** (`etl/`): daily land-cover files per state are validated, cleaned (each land type capped at its state's Census land area),
  converted to net CO2e with a factor for each state and land type taken from the EPA's 2022 state inventory (so a year of land
  cover at its baseline area adds up to EPA's real figure; forests are sinks, so totals are negative) and loaded into SQLite with a
  monthly state summary and a land-change log. Rerunning the pipeline replaces a period's rows instead of adding a second copy.
- **Shock detection:** each day is compared with the same state and land type's surrounding week, scaled by that series' own
  noise. It flags all 53 shocks the generator injected with no false alarms; the shocks (2–8% in a day) stand far above the 0.1%
  daily noise, so this confirms the method works rather than measuring how hard real data would be.
  The detector computes its rolling medians with grouped pandas operations instead of a Python function per series: **52% faster**
  on this project's 5,350 records (9.5 ms against 19.6 ms) and 24% faster on 273,750, flagging exactly the same days
  (`benchmarks/detection_time.py`).
- **Weekly preparation replaced** (`benchmarks/prep_hours.py`): over the 30 weeks of data, each week's work is counted from the
  pipeline's own steps (35 daily files, rows fixed, shocks to review, about 165 land changes to log, and the weekly summary, map
  layers, dashboard and forecast refresh) and costed at fixed per-step minutes for doing the same by hand (listed in the script).
  The median week is **10.2 hours** of manual preparation (9.8 to 10.8), which the pipeline does in about a second.
- **Forecasting on real data** (`models/epa_forecast.py`): the EPA's 25 annual series (5 states x 5 land types, 1990-2022) are
  forecast a year ahead from the three previous years, trained on years up to 2017 and scored on 2018-2022. Linear Regression
  reaches **94.2%** accuracy (1 − WAPE) and Random Forest **93.3%**, but repeating last year's value scores
  94.7%: land-use flux changes slowly, and predicting the year-to-year change instead does not beat that baseline either.
- **Forecasting the simulated daily balance** (`models/forecast.py`): a week ahead from lagged values only, scored on the final four
  weeks: Linear Regression 99.81%, Random Forest 99.83%, repeating last week's value 99.84%. Daily land
  cover barely moves outside shocks, so the models add nothing here. An earlier version reported R² = 1.0 because the target (area ×
  emission factor) could be computed from its own features.
- **Geospatial layer** (`etl/geo.py`) on US Census Bureau state boundaries (2023 cartographic file): outline areas are computed in
  an equal-area projection (EPSG:5070) and checked against Census land and water area; each state polygon carries its net CO2 and
  CO2 per km² of land; each detected shock gets a map point inside its state (the coordinates are synthetic, since the land-cover
  feed reports states, not locations), and a point-in-polygon spatial join confirms every point lies in its reported state. The
  state layer also carries the EPA's 2022 net land-use flux, its tons per km² of land, and the pipeline's yearly estimate next to it
  (within 0.2% of EPA for four states and 3.9% for Florida, whose net total is small). Both layers are written as GeoJSON for the
  ArcGIS Online dashboard.
- **Tests** cover the area cap (the Census land areas), the shock detector, the geospatial checks, the forecast features, the
  accuracy metric, the EPA extract and factors, the real-data forecast and rerunning a load.

## Architecture

```mermaid
flowchart LR
    E[etl/pipeline.py + transform.py<br/>clean, net CO2, shock detection] --> DB[(SQLite)]
    E --> G[etl/geo.py<br/>Census boundaries, spatial join] --> GJ[GeoJSON layers<br/>for ArcGIS Online]
    EPA[(EPA state inventory<br/>data/epa)] --> E
    EPA --> EF[models/epa_forecast.py<br/>year-ahead forecast vs baseline]
    EPA --> G
    DB --> F[models/forecast.py<br/>week-ahead forecast vs baseline]
    F --> X[exports/*.csv]
    DB --> V[viz/dashboard_export.py]
    X --> V
```

![Held-out forecast vs actual](viz/preview/co2_forecast_vs_actual.png)

## Stack

Python, pandas, NumPy, GeoPandas, Shapely, SQLite, scikit-learn, SQLAlchemy, Matplotlib, ArcGIS Online, pytest, GitHub Actions.

## Run it

```bash
pip install -r requirements.txt
python etl/pipeline.py
python models/forecast.py
python models/epa_forecast.py
python viz/dashboard_export.py
python benchmarks/prep_hours.py
python -m pytest tests/ -v
```

## Data

Daily land-cover files are generated per state by the pipeline (synthetic, seeded) when no source file is present: 5 states x
5 land types x 214 days (2025-04-01 to 2025-10-31), starting from baselines that roughly follow each state's land cover, with
seasonal swings, 0.1% daily noise and rare 2–8% shocks.

Real data: the EPA's *Inventory of U.S. Greenhouse Gas Emissions and Sinks by State: 1990-2022*, land-use sector, for the 5 states,
in `data/epa/lulucf_by_state.csv` (source, checksum and how it is built in `data/epa/SOURCE.md`; public domain). It sets the emission
factors, feeds the year-ahead forecast and is joined onto the state layer. State boundaries and land areas are real: US Census Bureau
`cb_2023_us_state_20m`, simplified to 0.01° and stored in `data/geo/states.geojson`.
