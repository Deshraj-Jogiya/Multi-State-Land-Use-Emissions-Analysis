"""Year-by-year backtest of the EPA land-use forecasts.

models/epa_forecast.py scores one split (train on target years up to 2017, test 2018-2022). One split can be luck, so this repeats
the test from eight forecast origins: for each target year 2015-2022, the models are trained on every earlier target year and
forecast that one year for all 25 series (5 states x 5 land types), each from the three real values before it.

Models: Linear Regression and Random Forest (the same features and settings as epa_forecast.py), and three simple baselines:
last year's value, the recent trend (last year plus last year's change) and the 3-year average. Accuracy is 1 - WAPE; skill is
1 - (model's absolute error / last year's value's absolute error), so above 0 means better than repeating last year.

Writes exports/epa_backtest_by_year.csv (accuracy and skill per model and target year), exports/epa_backtest_by_series.csv
(each state and land type over the eight years) and exports/epa_backtest_summary.csv (per model: mean, worst and best year, how
many years it beat last year's value, and the 10th-90th percentile of its yearly percentage errors as a range for a new forecast).
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
from models.epa_forecast import LAGS, accuracy, features  # noqa: E402

TARGETS = range(2015, 2023)
NAIVE = "Last year's value"


def predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = [f"lag{k}" for k in range(1, LAGS + 1)] + ["trend"]
    return {
        "Linear Regression": LinearRegression().fit(train[cols], train["net_tg_co2e"]).predict(test[cols]),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
        .fit(train[cols], train["net_tg_co2e"]).predict(test[cols]),
        NAIVE: test["lag1"].to_numpy(),
        "Recent trend": (test["lag1"] + test["trend"]).to_numpy(),
        "3-year average": test[["lag1", "lag2", "lag3"]].mean(axis=1).to_numpy(),
    }


def run(path: Path | None = None) -> pd.DataFrame:
    df = features(annual_tons(path) if path else annual_tons())
    rows = []
    for year in TARGETS:
        train, test = df[df["year"] < year], df[df["year"] == year]
        for name, pred in predictions(train, test).items():
            rows.append(test[["state", "land_type", "year", "net_tg_co2e"]].assign(model=name, predicted=pred))
    out = pd.concat(rows, ignore_index=True)
    out["abs_error"] = (out["net_tg_co2e"] - out["predicted"]).abs()
    return out


def summarise(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["state", "land_type", "year"]
    naive = preds.loc[preds["model"] == NAIVE, keys + ["abs_error"]].rename(columns={"abs_error": "naive_abs_error"})
    preds = preds.merge(naive, on=keys)

    def score(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"accuracy": accuracy(g["net_tg_co2e"].to_numpy(), g["predicted"].to_numpy()),
                          "skill_vs_last_year": 1 - g["abs_error"].sum() / g["naive_abs_error"].sum(), "points": len(g)})

    by_year = preds.groupby(["model", "year"]).apply(score, include_groups=False).reset_index()
    by_series = preds.groupby(["model", "state", "land_type"]).apply(score, include_groups=False).reset_index()
    # Percentage error of each yearly total (signed), to give a typical range for a new one-year-ahead forecast
    pct = preds.groupby(["model", "year"]).apply(
        lambda g: (g["predicted"] - g["net_tg_co2e"]).abs().sum() / g["net_tg_co2e"].abs().sum(), include_groups=False)
    summary = by_year.groupby("model").agg(mean_accuracy=("accuracy", "mean"), worst_year_accuracy=("accuracy", "min"),
                                           best_year_accuracy=("accuracy", "max"), mean_skill=("skill_vs_last_year", "mean"))
    summary["years_beating_last_year"] = by_year.assign(win=by_year["skill_vs_last_year"] > 0).groupby("model")["win"].sum()
    summary["wape_p10"] = pct.groupby("model").quantile(0.10)
    summary["wape_p90"] = pct.groupby("model").quantile(0.90)
    summary = summary.reset_index().sort_values("mean_accuracy", ascending=False)
    return by_year, by_series, summary


def main() -> pd.DataFrame:
    by_year, by_series, summary = summarise(run())
    out = ROOT / "exports"
    out.mkdir(exist_ok=True)
    by_year.round(4).to_csv(out / "epa_backtest_by_year.csv", index=False)
    by_series.round(4).to_csv(out / "epa_backtest_by_series.csv", index=False)
    summary.round(4).to_csv(out / "epa_backtest_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(main().round(4).to_string(index=False))
