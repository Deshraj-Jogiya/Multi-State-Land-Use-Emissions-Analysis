"""
etl/transform.py
----------------
Multi-State Land Use Emissions Analysis
Data cleaning, normalization, CO2 computation, outlier detection.

Author: Deshraj Jogiya
"""

import numpy as np
import pandas as pd
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] transform.py - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CO2_FACTORS: dict[str, float] = {
    "forest":      -3.5,   # tons CO2/ha  (sequestration)
    "urban":       +2.1,   # tons CO2/ha  (emission)
    "agriculture": +0.8,   # tons CO2/ha  (emission)
    "wetland":     -1.2,   # tons CO2/ha  (mild sequestration)
    "shrubland":   -0.3,   # tons CO2/ha  (mild sequestration)
}

VALID_STATES: set[str] = {"AL", "CA", "TX", "WA", "FL"}
VALID_LAND_TYPES: set[str] = set(CO2_FACTORS.keys())

AREA_MIN_HA: float = 0.0
AREA_MAX_HA: float = 5_000_000.0  # upper bound for sanity check
ANOMALY_Z_THRESHOLD: float = 2.5


# ---------------------------------------------------------------------------
# Step 1: Schema Validation
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """
    Ensures all required columns are present; drops rows missing critical fields.

    Required columns: state, date, land_type, area_ha
    """
    required_cols = {"state", "date", "land_type", "area_ha"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("[%s] Missing columns: %s — skipping file.", source, missing)
        return pd.DataFrame()

    before = len(df)
    df = df.dropna(subset=list(required_cols))
    dropped = before - len(df)
    if dropped:
        logger.info("[%s] Dropped %d rows with null required fields.", source, dropped)
    return df


# ---------------------------------------------------------------------------
# Step 2: State & Land-Type Filtering
# ---------------------------------------------------------------------------

def filter_valid_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes and filters state / land_type to known valid values.
    Unknown entries are dropped with a warning.
    """
    df = df.copy()
    df["state"]     = df["state"].astype(str).str.strip().str.upper()
    df["land_type"] = df["land_type"].astype(str).str.strip().str.lower()

    invalid_states = ~df["state"].isin(VALID_STATES)
    if invalid_states.any():
        logger.warning("Dropping %d rows with unrecognized states.", invalid_states.sum())
        df = df[~invalid_states]

    invalid_lt = ~df["land_type"].isin(VALID_LAND_TYPES)
    if invalid_lt.any():
        logger.warning("Dropping %d rows with unrecognized land types.", invalid_lt.sum())
        df = df[~invalid_lt]

    return df


# ---------------------------------------------------------------------------
# Step 3: Area Sanity Check
# ---------------------------------------------------------------------------

def clean_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts area_ha to numeric; clamps out-of-range values.
    Negative areas are set to 0; extreme outliers capped at AREA_MAX_HA.
    """
    df = df.copy()
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")

    # Fill NaN with column median per land_type
    df["area_ha"] = df.groupby("land_type")["area_ha"].transform(
        lambda x: x.fillna(x.median())
    )
    df["area_ha"] = df["area_ha"].fillna(df["area_ha"].median())

    # Clamp
    n_neg = (df["area_ha"] < AREA_MIN_HA).sum()
    n_big = (df["area_ha"] > AREA_MAX_HA).sum()
    if n_neg:
        logger.info("Clamping %d negative area_ha values to 0.", n_neg)
    if n_big:
        logger.info("Clamping %d oversized area_ha values to %.0f.", n_big, AREA_MAX_HA)

    df["area_ha"] = df["area_ha"].clip(lower=AREA_MIN_HA, upper=AREA_MAX_HA)
    return df


# ---------------------------------------------------------------------------
# Step 4: Date Normalization
# ---------------------------------------------------------------------------

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and normalizes the 'date' column to ISO-8601 (YYYY-MM-DD) strings.
    Rows with unparseable dates are dropped.
    """
    df = df.copy()
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unparseable dates.", dropped)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Step 5: CO2 Calculation
# ---------------------------------------------------------------------------

def compute_co2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates CO2 tons based on area_ha and per-land-type emission factor.
    Adds columns: emission_factor, co2_tons.
    """
    df = df.copy()
    df["emission_factor"] = df["land_type"].map(CO2_FACTORS)
    df["co2_tons"] = df["area_ha"] * df["emission_factor"]
    logger.info("Computed co2_tons for %d records.", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 6: Derived Features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features useful for ML and aggregation:
        year, month, day_of_year, week_of_year
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["year"]        = dates.dt.year
    df["month"]       = dates.dt.month
    df["day_of_year"] = dates.dt.day_of_year
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 7: Outlier Detection (Z-Score per land_type)
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, threshold: float = ANOMALY_Z_THRESHOLD) -> pd.DataFrame:
    """
    Flags rows where co2_tons is more than `threshold` standard deviations
    from the land_type group mean. Adds boolean column `is_anomaly`.
    """
    df = df.copy()
    df["is_anomaly"] = 0

    for lt, group in df.groupby("land_type"):
        mean  = group["co2_tons"].mean()
        std   = group["co2_tons"].std(ddof=1)
        if std == 0 or np.isnan(std):
            continue
        z_scores = (group["co2_tons"] - mean).abs() / std
        anomaly_idx = z_scores[z_scores > threshold].index
        df.loc[anomaly_idx, "is_anomaly"] = 1

    n_anomalies = df["is_anomaly"].sum()
    logger.info(
        "Outlier detection (z>%.1f): flagged %d / %d records (%.2f%%).",
        threshold, n_anomalies, len(df), 100 * n_anomalies / max(len(df), 1),
    )
    return df


# ---------------------------------------------------------------------------
# Step 8: Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicates on (state, date, land_type); keeps first occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["state", "date", "land_type"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info("Deduplicated %d exact duplicate rows.", dropped)
    return df


# ---------------------------------------------------------------------------
# Step 9: Source File Column
# ---------------------------------------------------------------------------

def ensure_source_file(df: pd.DataFrame, default: str = "synthetic") -> pd.DataFrame:
    """
    Ensures a 'source_file' column exists; fills missing values with `default`.
    """
    df = df.copy()
    if "source_file" not in df.columns:
        df["source_file"] = default
    else:
        df["source_file"] = df["source_file"].fillna(default)
    return df


# ---------------------------------------------------------------------------
# Master Transform Function
# ---------------------------------------------------------------------------

def transform(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """
    Full transformation pipeline. Applies all cleaning and enrichment steps in order.

    Parameters
    ----------
    df     : Raw DataFrame (from CSV or synthetic generator)
    source : Filename / label for logging

    Returns
    -------
    Cleaned, enriched DataFrame ready for database ingestion.
    """
    logger.info("=== Starting transform for source: %s | rows=%d ===", source, len(df))

    if df.empty:
        logger.warning("Empty DataFrame received for source: %s", source)
        return df

    df = validate_schema(df, source)
    if df.empty:
        return df

    df = filter_valid_categories(df)
    df = clean_area(df)
    df = normalize_dates(df)
    df = compute_co2(df)
    df = add_derived_features(df)
    df = detect_outliers(df)
    df = deduplicate(df)
    df = ensure_source_file(df, default=source)

    logger.info("=== Transform complete | output rows=%d ===", len(df))
    return df


# ---------------------------------------------------------------------------
# Land Change Detection
# ---------------------------------------------------------------------------

def detect_land_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares consecutive observations for the same state/land_type pair to
    identify significant area changes (>100 ha shift in 7 days).

    Returns a DataFrame matching the `land_changes` schema.
    """
    changes = []
    df_sorted = df.sort_values(["state", "land_type", "date"])

    for (state, land_type), group in df_sorted.groupby(["state", "land_type"]):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev_area = group.loc[i - 1, "area_ha"]
            curr_area = group.loc[i,     "area_ha"]
            delta     = curr_area - prev_area

            if abs(delta) > 100:  # threshold: 100 ha minimum change
                # Determine if this looks like a type conversion
                if delta < 0:
                    # Area shrinking → land leaving this type
                    from_type = land_type
                    # Assign gain to the next-largest type alphabetically (simplified heuristic)
                    other_types = [lt for lt in VALID_LAND_TYPES if lt != land_type]
                    to_type = sorted(other_types)[0]
                else:
                    to_type   = land_type
                    from_type = "forest"  # assume lost from forest (simplified)

                co2_factor_from = CO2_FACTORS.get(from_type, 0)
                co2_factor_to   = CO2_FACTORS.get(to_type,   0)
                co2_impact = abs(delta) * (co2_factor_to - co2_factor_from)

                changes.append({
                    "state":        state,
                    "change_date":  group.loc[i, "date"],
                    "from_type":    from_type,
                    "to_type":      to_type,
                    "area_changed": abs(delta),
                    "co2_impact":   co2_impact,
                    "notes":        f"Area shift {prev_area:.1f}→{curr_area:.1f} ha ({land_type})",
                })

    if not changes:
        logger.info("No significant land changes detected.")
        return pd.DataFrame(columns=["state", "change_date", "from_type", "to_type",
                                     "area_changed", "co2_impact", "notes"])

    change_df = pd.DataFrame(changes)
    logger.info("Detected %d land change events.", len(change_df))
    return change_df


# ---------------------------------------------------------------------------
# Normalization utility (for ML preprocessing)
# ---------------------------------------------------------------------------

def normalize_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces an ML-ready feature DataFrame with encoded categoricals and
    min-max normalized numeric columns.
    """
    df_ml = df[["state", "land_type", "area_ha", "co2_tons",
                "day_of_year", "month", "year"]].copy()

    # Label encode categoricals
    state_map = {s: i for i, s in enumerate(sorted(VALID_STATES))}
    lt_map    = {lt: i for i, lt in enumerate(sorted(VALID_LAND_TYPES))}

    df_ml["state_encoded"]     = df_ml["state"].map(state_map)
    df_ml["land_type_encoded"] = df_ml["land_type"].map(lt_map)

    # Min-max normalize area_ha
    area_min = df_ml["area_ha"].min()
    area_max = df_ml["area_ha"].max()
    if area_max > area_min:
        df_ml["area_ha_norm"] = (df_ml["area_ha"] - area_min) / (area_max - area_min)
    else:
        df_ml["area_ha_norm"] = 0.0

    return df_ml


if __name__ == "__main__":
    # Quick smoke test
    sample = pd.DataFrame({
        "state":     ["CA", "TX", "WA", "INVALID", "FL"],
        "date":      ["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "bad_date"],
        "land_type": ["forest", "urban", "wetland", "forest", "agriculture"],
        "area_ha":   [10000, 5000, -200, 3000, 8000],
    })
    result = transform(sample, source="smoke_test")
    print(result.to_string())
