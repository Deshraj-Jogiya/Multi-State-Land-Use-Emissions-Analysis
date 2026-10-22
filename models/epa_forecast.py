"""Forecast EPA's real state land-use emissions a year ahead.

The 25 series (5 states x 5 land types) of net flux from EPA's state inventory, 1990-2022 (data/epa/lulucf_by_state.csv), are
forecast one year ahead from the three previous years only. Models are trained on target years up to 2017 and scored on 2018-2022,
each prediction using the real values of the years before it. Both models are compared with repeating last year's value.
Accuracy is 1 - WAPE (sum of absolute errors over the sum of absolute actual values). Writes exports/epa_forecast_metrics.csv and
exports/epa_forecast_holdout.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from etl.epa_inventory import annual_tons  # noqa: E402

LAGS = 3
TEST_FROM = 2018


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - np.abs(actual - predicted).sum() / np.abs(actual).sum())


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["state", "land_type", "year"]).copy()
    g = df.groupby(["state", "land_type"])["net_tg_co2e"]
    for k in range(1, LAGS + 1):
        df[f"lag{k}"] = g.shift(k)
    df["trend"] = df["lag1"] - df["lag2"]
    return df.dropna()


def run(path: Path | None = None) -> dict:
    df = features(annual_tons(path) if path else annual_tons())
    cols = [f"lag{k}" for k in range(1, LAGS + 1)] + ["trend"]
    train, test = df[df["year"] < TEST_FROM], df[df["year"] >= TEST_FROM]
    preds = {
        "Linear Regression": LinearRegression().fit(train[cols], train["net_tg_co2e"]).predict(test[cols]),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
        .fit(train[cols], train["net_tg_co2e"]).predict(test[cols]),
        "Last year's value": test["lag1"].to_numpy(),
    }
    y = test["net_tg_co2e"].to_numpy()
    metrics = {name: round(accuracy(y, p), 4) for name, p in preds.items()}
    out = ROOT / "exports"
    out.mkdir(exist_ok=True)
    pd.DataFrame([{"model": k, "accuracy_1_minus_wape": v, "test_years": f"{TEST_FROM}-{int(test['year'].max())}",
                   "series": test.groupby(["state", "land_type"]).ngroups, "test_points": len(test)} for k, v in metrics.items()]
                 ).to_csv(out / "epa_forecast_metrics.csv", index=False)
    test[["state", "land_type", "year", "net_tg_co2e"]].assign(**{k: np.round(v, 4) for k, v in preds.items()}).to_csv(
        out / "epa_forecast_holdout.csv", index=False)
    return metrics


if __name__ == "__main__":
    for name, acc in run().items():
        print(f"{name:20s} {acc:.2%}")
