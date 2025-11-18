"""
etl/geo.py
----------
Geospatial layer for the five states, built on US Census Bureau cartographic boundaries
(cb_2023_us_state_20m, simplified to 0.01 degrees and stored in data/geo/states.geojson).

- Polygon areas are computed in an equal-area projection (EPSG:5070, CONUS Albers) and checked against the Census
  land + water figures, so a broken or mis-projected boundary is caught.
- Each detected shock gets a map location inside its state (synthetic: the land-cover feed reports states, not
  coordinates); a point-in-polygon spatial join then confirms every point falls inside the state it is reported for.
- outputs/geo/state_emissions.geojson: one polygon per state with net CO2, CO2 per km2 of land and shock counts.
- outputs/geo/shock_events.geojson: one point per detected shock.
These two layers feed the ArcGIS Online dashboard.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATES_PATH = ROOT / "data" / "geo" / "states.geojson"
OUT_DIR = ROOT / "outputs" / "geo"
EQUAL_AREA_CRS = "EPSG:5070"   # CONUS Albers; Washington through Florida all fall inside it
AREA_SLACK = 0.01              # 1% allowance for the 0.01-degree simplification


def load_states() -> gpd.GeoDataFrame:
    return gpd.read_file(STATES_PATH)


def check_areas(states: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Each outline's equal-area size must fall between the Census land area and land + water area: the cartographic
    outlines are clipped at the shoreline, so they include inland water but not all coastal and territorial water.
    """
    km2 = states.to_crs(EQUAL_AREA_CRS).geometry.area / 1e6
    land = states["census_land_m2"] / 1e6
    total = (states["census_land_m2"] + states["census_water_m2"]) / 1e6
    out = pd.DataFrame({"state": states["state"], "polygon_km2": km2.round(0), "census_land_km2": land.round(0),
                        "census_land_water_km2": total.round(0)})
    ok = (km2 >= land * (1 - AREA_SLACK)) & (km2 <= total * (1 + AREA_SLACK))
    if not ok.all():
        raise ValueError(f"State outlines disagree with Census area:\n{out[~ok]}")
    return out


def random_points_in(polygon, n: int, rng: np.random.Generator) -> list:
    """Uniform points inside a polygon by rejection sampling over its bounding box."""
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < n:
        xs, ys = rng.uniform(minx, maxx, n * 4), rng.uniform(miny, maxy, n * 4)
        points += [p for p in map(Point, xs, ys) if polygon.contains(p)]
    return points[:n]


def locate_shocks(records: pd.DataFrame, states: gpd.GeoDataFrame, seed: int = 7) -> gpd.GeoDataFrame:
    """Give each flagged shock a location in its state and confirm the placement with a spatial join."""
    shocks = records[records["is_anomaly"] == 1][["state", "date", "land_type", "area_ha", "co2_tons"]].copy()
    rng = np.random.default_rng(seed)
    shapes = states.set_index("state").geometry
    shocks["geometry"] = None
    for state, idx in shocks.groupby("state").groups.items():
        shocks.loc[idx, "geometry"] = random_points_in(shapes[state], len(idx), rng)
    events = gpd.GeoDataFrame(shocks, geometry="geometry", crs=states.crs)
    joined = gpd.sjoin(events, states[["state", "geometry"]].rename(columns={"state": "state_from_join"}),
                       how="left", predicate="within")
    mismatched = joined[joined["state"] != joined["state_from_join"]]
    if not mismatched.empty:
        raise ValueError(f"{len(mismatched)} shock locations fall outside their reported state")
    return events


def state_layer(records: pd.DataFrame, states: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    totals = (records.groupby("state")
                     .agg(days=("date", "nunique"), net_co2_tons=("co2_tons", "sum"), shocks=("is_anomaly", "sum"))
                     .reset_index())
    totals["avg_daily_net_co2_tons"] = (totals["net_co2_tons"] / totals["days"]).round(0)
    layer = states.merge(totals, on="state", how="left")
    layer["avg_daily_tons_per_km2_land"] = (layer["avg_daily_net_co2_tons"] / (layer["census_land_m2"] / 1e6)).round(3)
    return layer.drop(columns=["net_co2_tons"])


def build_layers(records: pd.DataFrame) -> dict:
    states = load_states()
    areas = check_areas(states)
    events = locate_shocks(records, states)
    layer = state_layer(records, states)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer.to_file(OUT_DIR / "state_emissions.geojson", driver="GeoJSON")
    events.assign(date=events["date"].astype(str)).to_file(OUT_DIR / "shock_events.geojson", driver="GeoJSON")
    logger.info("Geo layers written: %d states (outline areas within Census bounds), %d shock events, "
                "all inside their reported state by spatial join", len(layer), len(events))
    return {"states": layer, "events": events, "areas": areas}
