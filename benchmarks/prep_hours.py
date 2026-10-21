"""Estimate the weekly preparation time the pipeline replaces.

Runs the pipeline's own steps on every week of the data period and counts the work in each week: the daily files (5 states x 7
days), the rows each check had to fix, and the shocks to review. Each unit is costed with the per-step minutes in MANUAL_MINUTES
(what the same step takes by hand in a spreadsheet, set before running and not tuned to the result), and the pipeline's own run time
for the same week is measured. Writes exports/prep_hours.csv (one row per week) and prints the median week.
"""

import logging
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
logging.disable(logging.CRITICAL)
from etl.pipeline import END_DATE, RNG_SEED, START_DATE, STATES, generate_or_load_daily_file  # noqa: E402
from etl.transform import detect_land_changes, detect_outliers, transform  # noqa: E402

MANUAL_MINUTES = {
    "open_file": 3,          # download one state's daily file and open it
    "check_file": 3,         # check columns, state and land-type names, and areas against the state's land area
    "convert_file": 2,       # look up each land type's CO2 factor and compute net CO2
    "append_file": 2,        # paste the cleaned rows into the master table
    "fix_row": 3,            # correct one row a check rejected (missing value, unknown name, area out of range, duplicate)
    "review_shock": 10,      # compare a sudden change with the surrounding week and decide whether it is real
    "log_change": 1,         # record one day-to-day area shift over 100 ha with its CO2 impact in the land-change log
    "weekly_summary": 20,    # rebuild the per-state summary table
    "map_layers": 20,        # export the state and shock layers and upload them to the ArcGIS dashboard
    "dashboard": 15,         # refresh the dashboard charts
    "forecast": 30,          # refit the week-ahead forecast and compare it with last week's actuals
}
WEEKLY_TASKS = ("weekly_summary", "map_layers", "dashboard", "forecast")


def week_work(raw_files: list[pd.DataFrame], shocks: int, changes: int) -> dict:
    t0 = time.perf_counter()
    cleaned = [transform(f) for f in raw_files]
    seconds = time.perf_counter() - t0
    fixed = sum(len(r) - len(c) for r, c in zip(raw_files, cleaned)) + sum(
        int((pd.to_numeric(r["area_ha"], errors="coerce") != c["area_ha"].values).sum()) for r, c in zip(raw_files, cleaned)
        if len(r) == len(c))
    counts = {"files": len(raw_files), "rows_fixed": fixed, "shocks": shocks, "land_changes": changes}
    per_file = sum(MANUAL_MINUTES[k] for k in ("open_file", "check_file", "convert_file", "append_file"))
    minutes = (counts["files"] * per_file + counts["rows_fixed"] * MANUAL_MINUTES["fix_row"]
               + counts["shocks"] * MANUAL_MINUTES["review_shock"] + counts["land_changes"] * MANUAL_MINUTES["log_change"] + sum(MANUAL_MINUTES[k] for k in WEEKLY_TASKS))
    return {**counts, "manual_hours": round(minutes / 60, 2), "pipeline_seconds": round(seconds, 3)}


def main() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    days = []
    day = START_DATE
    while day <= END_DATE:  # same files, in the same order, as etl/pipeline.py
        days.append([generate_or_load_daily_file(s, day, rng) for s in STATES])
        day += timedelta(days=1)
    # shocks and land changes are found over the whole period, as the pipeline does, then counted in the week they fall in
    t0 = time.perf_counter()
    flagged = detect_outliers(pd.concat([transform(f) for d in days for f in d], ignore_index=True))
    changes = pd.to_datetime(detect_land_changes(flagged)["change_date"])
    detect_seconds = time.perf_counter() - t0
    flagged["date"] = pd.to_datetime(flagged["date"])
    rows = []
    for w in range(len(days) // 7):
        start = START_DATE + timedelta(days=7 * w)
        lo, hi = pd.Timestamp(start), pd.Timestamp(start + timedelta(days=6))
        in_week = flagged["date"].between(lo, hi)
        result = week_work([f for d in days[7 * w:7 * w + 7] for f in d], int(flagged.loc[in_week, "is_anomaly"].sum()),
                           int(changes.between(lo, hi).sum()))
        result["pipeline_seconds"] = round(result["pipeline_seconds"] + detect_seconds * 7 / len(days), 3)
        rows.append({"week_start": start.isoformat(), **result})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(BASE, "exports"), exist_ok=True)
    out.to_csv(os.path.join(BASE, "exports", "prep_hours.csv"), index=False)
    print(f"{len(out)} weeks: median {out['manual_hours'].median():.1f} hours of manual preparation a week "
          f"(range {out['manual_hours'].min():.1f}-{out['manual_hours'].max():.1f}), replaced by "
          f"{out['pipeline_seconds'].median():.2f} s of pipeline time")
    return out


if __name__ == "__main__":
    main()
