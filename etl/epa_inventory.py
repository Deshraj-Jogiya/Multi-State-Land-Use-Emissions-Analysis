"""EPA's state greenhouse-gas inventory for land use (real data).

Source: US EPA, Inventory of U.S. Greenhouse Gas Emissions and Sinks by State: 1990-2022 (EPA-430-R-24-006), consolidated workbook
allstateghgdata90-22_v082924.zip from https://www.epa.gov/ghgemissions/state-ghg-emissions-and-removals (sha256 in data/epa/SOURCE.md).
Sector "Land Use, Land-Use Change and Forestry", all gases in CO2 equivalent (AR5 100-year GWPs), Tg = million metric tons; emissions
are positive and removals (sinks) negative.

`extract` turns the workbook into the small table committed at data/epa/lulucf_by_state.csv: net flux per state, land type and year for
the pipeline's 5 states, with EPA's land categories mapped to the pipeline's land types. `annual_tons` reads that table back, and
`daily_factors` turns one year of it into tons CO2e per hectare per day for each state and land type, which compute_co2 uses in place of
illustrative factors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXTRACT = ROOT / "data" / "epa" / "lulucf_by_state.csv"
SECTOR = "Land Use, Land-Use Change and Forestry"
STATES = ["AL", "CA", "TX", "WA", "FL"]
# EPA land category -> the pipeline's land type
LAND_TYPES = {"Forest Land": "forest", "Cropland": "agriculture", "Grassland": "shrubland", "Settlements": "urban",
              "Wetlands": "wetland"}
FACTOR_YEAR = 2022


def extract(workbook: str | Path, out: Path = EXTRACT) -> pd.DataFrame:
    """Workbook -> long table (state, land_type, year, net_tg_co2e) for the 5 states, 1990-2022."""
    raw = pd.read_excel(workbook, sheet_name="Data by UNFCCC-IPCC Sectors")
    lu = raw[(raw["sector"] == SECTOR) & raw["geo_ref"].isin(STATES) & raw["subsector"].isin(LAND_TYPES)]
    years = [c for c in lu.columns if c.startswith("Y") and c[1:].isdigit()]
    wide = lu.groupby(["geo_ref", "subsector"])[years].sum().reset_index()
    long = wide.melt(id_vars=["geo_ref", "subsector"], var_name="year", value_name="net_tg_co2e")
    long = long.assign(state=long["geo_ref"], land_type=long["subsector"].map(LAND_TYPES), year=long["year"].str[1:].astype(int))
    long = long[["state", "land_type", "year", "net_tg_co2e"]].sort_values(["state", "land_type", "year"])
    long["net_tg_co2e"] = long["net_tg_co2e"].round(6)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False)
    return long


def annual_tons(path: Path = EXTRACT) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.assign(net_tons_co2e=df["net_tg_co2e"] * 1e6)


def daily_factors(baselines: dict[str, dict[str, float]], year: int = FACTOR_YEAR, path: Path = EXTRACT) -> dict[tuple[str, str], float]:
    """{(state, land_type): tons CO2e per hectare per day}: EPA's net flux for that year spread over the land type's baseline area."""
    df = annual_tons(path)
    df = df[df["year"] == year]
    return {(r.state, r.land_type): r.net_tons_co2e / baselines[r.state][r.land_type] / 365.0 for r in df.itertuples()}


if __name__ == "__main__":
    import sys

    table = extract(sys.argv[1])
    print(f"{len(table)} rows -> {EXTRACT}")
    print(table[table["year"] == FACTOR_YEAR].pivot(index="state", columns="land_type", values="net_tg_co2e").round(2))
