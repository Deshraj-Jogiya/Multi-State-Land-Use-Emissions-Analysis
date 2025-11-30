"""The year-by-year backtest scores every model on every series for all eight target years, never with future data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import epa_backtest  # noqa: E402


def test_every_model_forecasts_all_25_series_for_each_of_the_8_years():
    preds = epa_backtest.run()
    assert sorted(preds["year"].unique()) == list(epa_backtest.TARGETS)
    assert preds.groupby(["model", "year"]).size().eq(25).all()
    assert preds["model"].nunique() == 5


def test_last_years_value_is_the_reference_for_skill():
    by_year, _, summary = epa_backtest.summarise(epa_backtest.run())
    naive = summary.set_index("model").loc[epa_backtest.NAIVE]
    assert naive["mean_skill"] == 0 and naive["years_beating_last_year"] == 0
    assert by_year["accuracy"].between(0, 1).all()
