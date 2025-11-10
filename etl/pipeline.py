"""
etl/pipeline.py
---------------
Multi-State Land Use Emissions Analysis
Main ETL pipeline — ingests (or generates) daily land cover CSVs for 5 U.S. states,
transforms data, and loads results into a SQLite database.

Run:  python etl/pipeline.py

Author: Deshraj Jogiya
"""

import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from etl.transform import transform, detect_land_changes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] pipeline.py - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATES: list[str]       = ["AL", "CA", "TX", "WA", "FL"]
LAND_TYPES: list[str]   = ["forest", "urban", "agriculture", "wetland", "shrubland"]

START_DATE: date = date(2025, 4, 1)
END_DATE:   date = date(2025, 10, 31)          # ~213 days of data

DATA_DIR:   Path = ROOT / "data" / "raw"
OUTPUT_DIR: Path = ROOT / "outputs"
DB_PATH:    Path = OUTPUT_DIR / "land_use_emissions.db"

SCHEMA_PATH: Path = ROOT / "db" / "schema.sql"

# CO2 factors (tons CO2 per hectare; negative = sequestration)
CO2_FACTORS: dict[str, float] = {
    "forest":      -3.5,
    "urban":       +2.1,
    "agriculture": +0.8,
    "wetland":     -1.2,
    "shrubland":   -0.3,
}

# Realistic baseline areas per state (hectares) — sum over land types
# Proportions roughly reflect actual state land cover distributions
STATE_BASELINES: dict[str, dict[str, float]] = {
    "AL": {"forest": 9_500_000, "urban":   620_000, "agriculture": 3_200_000, "wetland":  400_000, "shrubland":  180_000},
    "CA": {"forest": 6_800_000, "urban": 2_400_000, "agriculture": 9_600_000, "wetland":  150_000, "shrubland": 4_200_000},
    "TX": {"forest": 5_600_000, "urban": 2_900_000, "agriculture":19_000_000, "wetland":  800_000, "shrubland": 5_300_000},
    "WA": {"forest":10_400_000, "urban":   720_000, "agriculture": 4_200_000, "wetland":  320_000, "shrubland": 1_100_000},
    "FL": {"forest": 6_100_000, "urban": 1_800_000, "agriculture": 4_400_000, "wetland": 2_900_000, "shrubland":  400_000},
}

# Random seed for reproducibility
RNG_SEED: int = 42


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_csv(state: str, run_date: date, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generates a synthetic land cover observation file for one state on one date.

    Simulates realistic daily variation:
    - Small random walk on area (±0.1% per day) layered with seasonal signal.
    - Occasional large shocks to simulate forest fires / urban expansion events.

    Returns a DataFrame with columns: state, date, land_type, area_ha, source_file
    """
    day_of_year = run_date.timetuple().tm_yday

    rows = []
    baseline = STATE_BASELINES[state]

    for land_type in LAND_TYPES:
        base_area = baseline[land_type]

        # Seasonal multiplier: urban peaks in summer, forest dips (fire season)
        seasonal_factor = 1.0
        if land_type == "urban":
            seasonal_factor = 1 + 0.005 * np.sin(2 * np.pi * day_of_year / 365)
        elif land_type == "forest":
            seasonal_factor = 1 - 0.008 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
        elif land_type == "agriculture":
            seasonal_factor = 1 + 0.015 * np.cos(2 * np.pi * day_of_year / 365)

        # Random walk noise (±0.1% daily)
        noise_pct = rng.normal(0, 0.001)

        # Rare shock events (~1% probability): simulate wildfire or urban sprawl
        shock = 0.0
        if rng.random() < 0.01:
            shock = rng.choice([-1, 1]) * rng.uniform(0.02, 0.08)  # ±2–8% shock
            logger.debug("  [%s] Shock event on %s for %s: %.2f%%", state, run_date, land_type, shock * 100)

        area_ha = max(0.0, base_area * seasonal_factor * (1 + noise_pct + shock))

        rows.append({
            "state":       state,
            "date":        run_date.isoformat(),
            "land_type":   land_type,
            "area_ha":     round(area_ha, 2),
            "source_file": f"{state}_{run_date.isoformat()}_landcover.csv",
        })

    return pd.DataFrame(rows)


def generate_or_load_daily_file(state: str, run_date: date, rng: np.random.Generator) -> pd.DataFrame:
    """
    Attempts to load a real CSV file; generates synthetic data if not found.
    CSV path pattern: data/raw/<STATE>/<STATE>_<YYYY-MM-DD>_landcover.csv
    """
    csv_path = DATA_DIR / state / f"{state}_{run_date.isoformat()}_landcover.csv"

    if csv_path.exists():
        logger.debug("Loading existing file: %s", csv_path)
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        return df
    else:
        df = generate_synthetic_csv(state, run_date, rng)
        return df


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def initialize_database(engine) -> None:
    """
    Executes schema.sql to create all tables if they don't exist.
    Reads the schema file and applies each CREATE TABLE statement.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SCHEMA_PATH.exists():
        logger.info("Applying schema from %s", SCHEMA_PATH)
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        # Split on semicolons and execute non-empty statements
        with engine.connect() as conn:
            for stmt in schema_sql.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        conn.execute(text(stmt))
                    except Exception as exc:
                        logger.debug("Schema stmt skipped (%s): %.80s", exc, stmt)
            conn.commit()
    else:
        # Inline fallback schema if schema.sql is missing
        logger.warning("schema.sql not found — creating minimal inline schema.")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS land_use_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    state           TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    land_type       TEXT NOT NULL,
                    area_ha         REAL NOT NULL,
                    co2_tons        REAL NOT NULL,
                    emission_factor REAL NOT NULL,
                    source_file     TEXT NOT NULL DEFAULT 'synthetic',
                    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
                    is_anomaly      INTEGER NOT NULL DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS land_changes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    state           TEXT NOT NULL,
                    change_date     TEXT NOT NULL,
                    from_type       TEXT NOT NULL,
                    to_type         TEXT NOT NULL,
                    area_changed    REAL NOT NULL,
                    co2_impact      REAL NOT NULL,
                    notes           TEXT DEFAULT NULL
                )
            """))
            conn.commit()

    logger.info("Database initialized: %s", DB_PATH)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_records(df: pd.DataFrame, engine, table: str = "land_use_records") -> int:
    """
    Appends cleaned records to the target table.
    Returns the number of rows inserted.
    """
    cols_for_table = [
        "state", "date", "land_type", "area_ha",
        "co2_tons", "emission_factor", "source_file", "is_anomaly",
    ]
    # Only keep columns that exist in df
    cols = [c for c in cols_for_table if c in df.columns]
    df_insert = df[cols].copy()
    df_insert.to_sql(table, con=engine, if_exists="append", index=False)
    return len(df_insert)


def ingest_changes(change_df: pd.DataFrame, engine) -> int:
    """
    Appends land change events to the land_changes table.
    """
    if change_df.empty:
        return 0
    cols = ["state", "change_date", "from_type", "to_type", "area_changed", "co2_impact", "notes"]
    change_df[cols].to_sql("land_changes", con=engine, if_exists="append", index=False)
    return len(change_df)


# ---------------------------------------------------------------------------
# State Monthly Summary (materialization)
# ---------------------------------------------------------------------------

def update_state_summary(engine) -> None:
    """
    Recomputes the state_summary table from land_use_records.
    Uses upsert via DELETE+INSERT for simplicity with SQLite.
    """
    logger.info("Updating state_summary table ...")
    query = """
        SELECT
            state,
            CAST(strftime('%Y', date) AS INTEGER) AS year,
            CAST(strftime('%m', date) AS INTEGER) AS month,
            SUM(area_ha)  AS total_area_ha,
            SUM(co2_tons) AS total_co2_tons,
            COUNT(*)      AS record_count
        FROM land_use_records
        GROUP BY state, year, month
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        cols = result.keys()

    summary_df = pd.DataFrame(rows, columns=cols)

    # Find dominant land type per state-month
    dom_query = """
        SELECT state,
               CAST(strftime('%Y', date) AS INTEGER) AS year,
               CAST(strftime('%m', date) AS INTEGER) AS month,
               land_type,
               SUM(area_ha) AS area
        FROM land_use_records
        GROUP BY state, year, month, land_type
    """
    with engine.connect() as conn:
        dom_result = conn.execute(text(dom_query))
        dom_rows   = dom_result.fetchall()
        dom_cols   = dom_result.keys()

    dom_df = pd.DataFrame(dom_rows, columns=dom_cols)
    dominant = (
        dom_df.sort_values("area", ascending=False)
              .groupby(["state", "year", "month"])["land_type"]
              .first()
              .reset_index()
              .rename(columns={"land_type": "dominant_land"})
    )
    summary_df = summary_df.merge(dominant, on=["state", "year", "month"], how="left")
    summary_df["dominant_land"] = summary_df["dominant_land"].fillna("unknown")

    summary_df.to_sql("state_summary", con=engine, if_exists="replace", index=False)
    logger.info("state_summary updated with %d rows.", len(summary_df))


# ---------------------------------------------------------------------------
# Main ETL Orchestration
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Full ETL run:
      1. Generate date range
      2. For each state × date: load or generate CSV, transform, buffer
      3. Bulk ingest to SQLite
      4. Detect and ingest land changes
      5. Refresh summary table
    """
    pipeline_start = time.time()
    logger.info("=" * 65)
    logger.info("Multi-State Land Use Emissions Analysis — ETL Pipeline")
    logger.info("States: %s  |  Period: %s → %s", STATES, START_DATE, END_DATE)
    logger.info("=" * 65)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    initialize_database(engine)

    rng = np.random.default_rng(RNG_SEED)

    # Build date range
    date_range: list[date] = []
    current = START_DATE
    while current <= END_DATE:
        date_range.append(current)
        current += timedelta(days=1)

    total_days   = len(date_range)
    logger.info("Processing %d days × %d states = %d total date-state pairs.",
                total_days, len(STATES), total_days * len(STATES))

    all_records: list[pd.DataFrame] = []
    files_processed = 0

    for run_date in date_range:
        for state in STATES:
            raw_df = generate_or_load_daily_file(state, run_date, rng)
            cleaned_df = transform(raw_df, source=raw_df["source_file"].iloc[0] if not raw_df.empty else "unknown")
            if not cleaned_df.empty:
                all_records.append(cleaned_df)
            files_processed += 1

        # Progress log every 30 days
        if run_date.day == 1:
            logger.info("Progress: %s processed (files=%d)", run_date, files_processed)

    if not all_records:
        logger.error("No records generated — aborting.")
        return

    # Combine all daily records
    combined_df = pd.concat(all_records, ignore_index=True)
    logger.info("Total records before ingestion: %d", len(combined_df))

    # Ingest land_use_records
    n_inserted = ingest_records(combined_df, engine)
    logger.info("Inserted %d records into land_use_records.", n_inserted)

    # Detect and ingest land changes
    logger.info("Running land change detection ...")
    change_df = detect_land_changes(combined_df)
    n_changes = ingest_changes(change_df, engine)
    logger.info("Inserted %d rows into land_changes.", n_changes)

    # Refresh summary
    update_state_summary(engine)

    # Final report
    elapsed = time.time() - pipeline_start
    logger.info("=" * 65)
    logger.info("ETL COMPLETE in %.1f seconds.", elapsed)
    logger.info("  Records inserted : %d", n_inserted)
    logger.info("  Land changes     : %d", n_changes)
    logger.info("  Database path    : %s", DB_PATH)
    logger.info("=" * 65)

    # Print quick summary to stdout
    summary_query = """
        SELECT state,
               ROUND(SUM(co2_tons), 0) AS total_co2_tons,
               COUNT(*) AS records
        FROM land_use_records
        GROUP BY state
        ORDER BY total_co2_tons DESC
    """
    with engine.connect() as conn:
        result = conn.execute(text(summary_query))
        rows   = result.fetchall()

    print("\n--- Emission Summary by State ---")
    print(f"{'State':<6} {'Total CO2 (tons)':>18} {'Records':>10}")
    print("-" * 38)
    for row in rows:
        state, co2, cnt = row
        print(f"{state:<6} {co2:>18,.0f} {cnt:>10,}")
    print()


if __name__ == "__main__":
    run_pipeline()
