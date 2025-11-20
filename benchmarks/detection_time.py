"""Time the shock detector against the earlier version, which ran a Python function per state and land-type series.

Both run on simulated series shaped like the pipeline's (5 land types per state, 0.1% daily noise, 1% of days with a 2-8% shock):
the project's own size (5 states x 214 days) and a larger one (50 states x 3 years). Checks both flag the same days, takes the best
of 5 runs, and writes exports/detection_benchmark.csv.
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from etl.transform import SHOCK_THRESHOLD, SHOCK_WINDOW_DAYS, detect_outliers  # noqa: E402

LAND_TYPES = ["forest", "wetland", "agriculture", "shrubland", "urban"]


def detect_outliers_per_series(df: pd.DataFrame, threshold: float = SHOCK_THRESHOLD) -> pd.DataFrame:
    """The earlier implementation: a Python lambda per series inside groupby().transform()."""
    df = df.copy().sort_values(["state", "land_type", "date"])
    rel = df.groupby(["state", "land_type"])["area_ha"].transform(
        lambda x: x / x.rolling(SHOCK_WINDOW_DAYS, center=True, min_periods=3).median() - 1)
    mad = rel.groupby([df["state"], df["land_type"]]).transform(lambda r: (r - r.median()).abs().median())
    robust_z = (rel - rel.groupby([df["state"], df["land_type"]]).transform("median")).abs() / (1.4826 * mad)
    df["is_anomaly"] = (robust_z > threshold).fillna(False).astype(int)
    return df.sort_index()


def simulated(states: int, days: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days)
    frames = []
    for s in range(states):
        for land_type in LAND_TYPES:
            area = rng.uniform(1e5, 1e6) * (1 + rng.normal(0, 0.001, days))
            hit = rng.random(days) < 0.01
            area[hit] *= 1 + rng.uniform(0.02, 0.08, hit.sum())
            frames.append(pd.DataFrame({"state": f"S{s}", "land_type": land_type, "date": dates, "area_ha": area}))
    return pd.concat(frames, ignore_index=True)


def best_of(fn, df, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter(); fn(df); times.append(time.perf_counter() - start)
    return min(times)


if __name__ == "__main__":
    logging.disable(logging.INFO)
    rows = []
    for states, days in [(5, 214), (50, 1095)]:
        df = simulated(states, days)
        assert detect_outliers(df)["is_anomaly"].equals(detect_outliers_per_series(df)["is_anomaly"])
        before, after = best_of(detect_outliers_per_series, df), best_of(detect_outliers, df)
        rows.append({"states": states, "days": days, "records": len(df), "per_series_ms": round(before * 1000, 1),
                     "vectorised_ms": round(after * 1000, 1), "time_saved": round(1 - after / before, 4)})
    os.makedirs(os.path.join(BASE, "exports"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(BASE, "exports", "detection_benchmark.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
