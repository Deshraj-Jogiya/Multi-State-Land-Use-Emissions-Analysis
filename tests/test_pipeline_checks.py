import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl import transform as tf  # noqa: E402
from models import forecast as fc  # noqa: E402


def _series(values, state="AL", land_type="forest"):
    dates = pd.date_range("2025-04-01", periods=len(values), freq="D")
    return pd.DataFrame({"state": state, "land_type": land_type, "date": dates.strftime("%Y-%m-%d"),
                         "area_ha": values})


def test_area_cap_is_the_states_land_area_not_a_flat_five_million():
    df = _series([9_500_000.0, 70_000_000.0])
    out = tf.clean_area(df)
    assert out["area_ha"].iloc[0] == 9_500_000.0          # Alabama's forest is kept as is
    assert out["area_ha"].iloc[1] == tf.STATE_LAND_AREA_HA["AL"]  # larger than the whole state: capped


def test_a_one_day_shock_is_flagged_and_ordinary_noise_is_not():
    rng = np.random.default_rng(0)
    values = 1_000_000 * (1 + rng.normal(0, 0.001, 60))
    values[30] *= 1.05
    flagged = tf.detect_outliers(_series(values))["is_anomaly"]
    assert flagged.iloc[30] == 1 and flagged.sum() == 1


def test_forecast_features_exclude_the_same_day_area():
    assert "area_ha" not in fc.FEATURE_COLS and "area_x_factor" not in fc.FEATURE_COLS
    df = _series(np.arange(30, dtype=float) + 100).assign(co2_tons=lambda d: d["area_ha"] * -3.5)
    feats = fc.prepare_features(df)
    last = feats.iloc[-1]
    assert last["lag_7_co2"] == df["co2_tons"].iloc[-8]


def test_forecast_accuracy_uses_magnitudes():
    assert fc.forecast_accuracy([-100, 100], [-90, 110]) == 0.9


def test_state_outlines_match_census_and_shock_points_land_in_their_state():
    from etl import geo
    states = geo.load_states()
    assert set(states["state"]) == {"AL", "CA", "TX", "WA", "FL"}
    geo.check_areas(states)  # raises if an outline's area is outside the Census land / land+water bounds
    records = pd.DataFrame({"state": ["WA", "FL", "TX"], "date": ["2025-05-01"] * 3, "land_type": ["forest"] * 3,
                            "area_ha": [1.0] * 3, "co2_tons": [-3.5] * 3, "is_anomaly": [1, 1, 1]})
    events = geo.locate_shocks(records, states)  # raises if any point falls outside its state
    assert len(events) == 3


def test_cap_values_are_the_census_land_areas():
    from etl import geo
    states = geo.load_states().set_index("state")
    for state, cap in tf.STATE_LAND_AREA_HA.items():
        assert abs(cap - states.loc[state, "census_land_m2"] / 1e4) < 1
