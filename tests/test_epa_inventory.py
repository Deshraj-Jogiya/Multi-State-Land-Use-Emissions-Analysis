"""The EPA state inventory extract, the factors derived from it, the real-data forecast and rerun safety of the database load."""
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl import epa_inventory as epa  # noqa: E402
from etl import pipeline  # noqa: E402
from etl import transform as tf  # noqa: E402
from models import epa_forecast  # noqa: E402


def test_extract_covers_5_states_5_land_types_and_33_years():
    df = epa.annual_tons()
    assert len(df) == 5 * 5 * 33
    assert set(df["state"]) == set(epa.STATES) and set(df["land_type"]) == set(epa.LAND_TYPES.values())
    assert (df["year"].min(), df["year"].max()) == (1990, 2022)
    # spot values from EPA's workbook (Tg CO2e, 2022): Alabama's forests are a sink, Florida's cropland a source
    al_forest = df[(df.state == "AL") & (df.land_type == "forest") & (df.year == 2022)]["net_tg_co2e"].item()
    fl_crop = df[(df.state == "FL") & (df.land_type == "agriculture") & (df.year == 2022)]["net_tg_co2e"].item()
    assert round(al_forest, 2) == -52.51 and round(fl_crop, 2) == 13.02


def test_a_year_at_baseline_area_adds_up_to_epas_figure():
    df = epa.annual_tons()
    real = df[df.year == epa.FACTOR_YEAR].set_index(["state", "land_type"])["net_tons_co2e"]
    for (state, land_type), factor in tf.EMISSION_FACTORS.items():
        assert np.isclose(factor * tf.STATE_BASELINES[state][land_type] * 365, real[(state, land_type)])


def test_compute_co2_uses_the_states_own_factor():
    df = pd.DataFrame({"state": ["AL", "WA"], "land_type": ["forest", "forest"], "area_ha": [1.0, 1.0]})
    out = tf.compute_co2(df)
    assert out["emission_factor"].tolist() == [tf.EMISSION_FACTORS[("AL", "forest")], tf.EMISSION_FACTORS[("WA", "forest")]]
    assert out["emission_factor"].iloc[0] != out["emission_factor"].iloc[1]


def test_real_data_forecast_runs_and_reports_the_baseline():
    metrics = epa_forecast.run()
    assert set(metrics) == {"Linear Regression", "Random Forest", "Last year's value"}
    assert all(0.8 < v <= 1 for v in metrics.values())


def test_loading_the_same_period_twice_keeps_one_copy(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    rows = pd.DataFrame({"state": ["AL", "AL"], "date": ["2025-04-01", "2025-04-02"], "land_type": ["forest", "forest"],
                         "area_ha": [1.0, 2.0], "co2_tons": [-1.0, -2.0], "emission_factor": [-1.0, -1.0],
                         "source_file": ["x", "x"], "is_anomaly": [0, 0]})
    changes = pd.DataFrame({"state": ["AL"], "change_date": ["2025-04-02"], "from_type": ["forest"], "to_type": ["urban"],
                            "area_changed": [1.0], "co2_impact": [0.1], "notes": [""]})
    for _ in range(2):
        pipeline.ingest_records(rows, engine)
        pipeline.ingest_changes(changes, engine)
    with engine.connect() as conn:
        assert pd.read_sql("SELECT COUNT(*) AS n FROM land_use_records", conn)["n"].item() == 2
        assert pd.read_sql("SELECT COUNT(*) AS n FROM land_changes", conn)["n"].item() == 1
