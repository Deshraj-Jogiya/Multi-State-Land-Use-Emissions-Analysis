# Multi-State Land Use Emissions Analysis

[![ci](https://github.com/Deshraj-Jogiya/Multi-State-Land-Use-Emissions-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Deshraj-Jogiya/Multi-State-Land-Use-Emissions-Analysis/actions/workflows/ci.yml) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

An ETL and analysis pipeline over daily land-cover data for 5 U.S. states (Alabama, California, Texas, Washington, Florida): it builds
a daily net CO2 balance per state and land type, flags sudden land-cover shocks such as wildfires or bursts of urban expansion, and
tests whether Random Forest and Linear Regression can forecast the balance a week ahead.

## Highlights

- **ETL** (`etl/`): daily land-cover files per state are validated, cleaned (each land type capped at its state's Census land area),
  converted to net CO2 with per-land-type factors (forests and wetlands are sinks, so totals are negative) and loaded into SQLite
  with a monthly state summary and a land-change log.
- **Shock detection:** each day is compared with the same state and land type's surrounding week, scaled by that series' own
  noise. It flags all 53 shocks the generator injected with no false alarms; the shocks (2–8% in a day) stand far above the 0.1%
  daily noise, so this confirms the method works rather than measuring how hard real data would be.
- **Forecasting, measured honestly:** both models forecast each state and land type a week ahead from lagged values only and are
  scored on the final four weeks. Random Forest reaches 99.79% accuracy (1 − WAPE) and Linear Regression 99.77%, but repeating
  last week's value already reaches 99.82%: daily land cover barely moves outside shocks, so the models add nothing here. An
  earlier version reported R² = 1.0 because the target (area × emission factor) could be computed from its own features.
- **Tests** cover the area cap, the shock detector, the forecast features and the accuracy metric.

## Architecture

```mermaid
flowchart LR
    E[etl/pipeline.py + transform.py<br/>clean, net CO2, shock detection] --> DB[(SQLite)]
    DB --> F[models/forecast.py<br/>week-ahead forecast vs baseline]
    F --> X[exports/*.csv]
    DB --> V[viz/dashboard_export.py]
    X --> V
```

![Held-out forecast vs actual](viz/preview/co2_forecast_vs_actual.png)

## Stack

Python, pandas, NumPy, SQLite, scikit-learn, SQLAlchemy, Matplotlib, pytest, GitHub Actions.

## Run it

```bash
pip install -r requirements.txt
python etl/pipeline.py
python models/forecast.py
python viz/dashboard_export.py
python -m pytest tests/ -v
```

## Data

Daily land-cover files are generated per state by the pipeline (synthetic, seeded) when no source file is present: 5 states x
5 land types x 214 days (2025-04-01 to 2025-10-31), starting from baselines that roughly follow each state's land cover, with
seasonal swings, 0.1% daily noise and rare 2–8% shocks. The CO2 factors are illustrative, not an official inventory.
