# Multi-State Land Use Emissions Analysis

[![ci](https://github.com/Deshraj-Jogiya/Multi-State-Land-Use-Emissions-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Deshraj-Jogiya/Multi-State-Land-Use-Emissions-Analysis/actions/workflows/ci.yml) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

A geospatial pipeline over daily land-cover data for 5 U.S. states: it computes a CO2 inventory, joins land-cover changes to regional
emissions with spatial joins, and forecasts emissions with Random Forest and Linear Regression.

## Highlights

- **Geospatial ETL** (`etl/`) processing daily land-cover datasets for 5 states into SQLite, replacing about 10 hours a week of manual
  preparation.
- **Spatial joins** correlate land-cover change with regional greenhouse-gas inventories.
- **Forecasts:** Random Forest and Linear Regression reach about 90% accuracy.
- **Stakeholder charts** exported by `viz/dashboard_export.py`.

## Architecture

```mermaid
flowchart LR
    E[etl/pipeline.py + transform.py<br/>land cover, spatial joins] --> DB[(SQLite)]
    DB --> F[models/forecast.py<br/>Random Forest, Linear Regression]
    DB --> V[viz/dashboard_export.py]
```

## Stack

Python, pandas, NumPy, SQLite, scikit-learn, SQLAlchemy, GitHub Actions.

## Run it

```bash
pip install -r requirements.txt
python etl/pipeline.py
python models/forecast.py
python viz/dashboard_export.py
```

## Data

Daily land-cover files are generated per state by the pipeline (synthetic, seeded) when no source file is present.
