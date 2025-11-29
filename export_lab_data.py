"""Exports EPA's state land-use series and the forecast models' holdout predictions for the portfolio Lab demo.

Usage: python export_lab_data.py <output_dir>

The demo shows any of the 25 series (5 states x 5 land types, 1990-2022) with the 2018-2022 forecasts. The browser refits the linear
regression itself from the series; the Random Forest's predictions come from this file, since a 300-tree forest is not rebuilt in
the page. This writes emissions.json: each series' years and values (Tg CO2e), every holdout year's prediction from each model in
models/epa_forecast.py, and the accuracies in exports/epa_forecast_metrics.csv, which the page's check compares against.
"""
import json
import os
import sys

import pandas as pd

from models import epa_forecast

BASE = os.path.dirname(os.path.abspath(__file__))


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics = epa_forecast.run()
    raw = pd.read_csv(os.path.join(BASE, "data", "epa", "lulucf_by_state.csv"))
    holdout = pd.read_csv(os.path.join(BASE, "exports", "epa_forecast_holdout.csv"))
    series = []
    for (state, land), g in raw.sort_values("year").groupby(["state", "land_type"]):
        h = holdout[(holdout.state == state) & (holdout.land_type == land)].sort_values("year")
        series.append({"state": state, "land_type": land, "years": g.year.astype(int).tolist(),
                       "values": [round(float(v), 6) for v in g.net_tg_co2e],
                       "holdout": {m: [round(float(v), 4) for v in h[m]] for m in metrics}})
    with open(os.path.join(out_dir, "emissions.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "EPA, Inventory of U.S. Greenhouse Gas Emissions and Sinks by State: 1990-2022 (EPA-430-R-24-006); "
                             "net land-use flux, Tg CO2e, emissions positive and removals negative",
                   "lags": epa_forecast.LAGS, "test_from": epa_forecast.TEST_FROM, "accuracy": metrics, "series": series},
                  f, separators=(",", ":"))
    print(f"{len(series)} series written to {out_dir}: {metrics}")


if __name__ == "__main__":
    main(sys.argv[1])
