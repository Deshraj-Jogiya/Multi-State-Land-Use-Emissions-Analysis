-- =============================================================================
-- Multi-State Land Use Emissions Analysis
-- Database Schema
-- Author: Deshraj Jogiya
-- =============================================================================

-- Drop tables if they exist (for clean re-runs)
DROP TABLE IF EXISTS land_use_records;
DROP TABLE IF EXISTS land_changes;
DROP TABLE IF EXISTS state_summary;
DROP TABLE IF EXISTS anomaly_flags;

-- ---------------------------------------------------------------------------
-- Table: land_use_records
-- Primary fact table storing daily land cover observations per state.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS land_use_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    state           TEXT    NOT NULL CHECK(state IN ('AL','CA','TX','WA','FL')),
    date            TEXT    NOT NULL,                        -- ISO-8601 date: YYYY-MM-DD
    land_type       TEXT    NOT NULL CHECK(land_type IN ('forest','urban','agriculture','wetland','shrubland')),
    area_ha         REAL    NOT NULL CHECK(area_ha >= 0),    -- Hectares
    co2_tons        REAL    NOT NULL,                        -- Positive = emission, negative = sequestration
    emission_factor REAL    NOT NULL,                        -- tons CO2 per hectare (land_type constant)
    source_file     TEXT    NOT NULL DEFAULT 'synthetic',    -- Originating CSV filename
    ingested_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    is_anomaly      INTEGER NOT NULL DEFAULT 0               -- 0=normal, 1=anomaly (>2 std dev)
);

-- Index for fast time-series queries
CREATE INDEX IF NOT EXISTS idx_lur_state_date ON land_use_records (state, date);
CREATE INDEX IF NOT EXISTS idx_lur_date       ON land_use_records (date);
CREATE INDEX IF NOT EXISTS idx_lur_land_type  ON land_use_records (land_type);
CREATE INDEX IF NOT EXISTS idx_lur_anomaly    ON land_use_records (is_anomaly);

-- ---------------------------------------------------------------------------
-- Table: land_changes
-- Tracks land-use type transitions (e.g., forest → urban).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS land_changes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    state           TEXT    NOT NULL CHECK(state IN ('AL','CA','TX','WA','FL')),
    change_date     TEXT    NOT NULL,                        -- ISO-8601 date
    from_type       TEXT    NOT NULL,                        -- Original land type
    to_type         TEXT    NOT NULL,                        -- New land type
    area_changed    REAL    NOT NULL CHECK(area_changed > 0),-- Hectares converted
    co2_impact      REAL    NOT NULL,                        -- Net CO2 impact of transition
    notes           TEXT    DEFAULT NULL                     -- Optional description
);

-- Index for change analysis
CREATE INDEX IF NOT EXISTS idx_lc_state       ON land_changes (state);
CREATE INDEX IF NOT EXISTS idx_lc_change_date ON land_changes (change_date);
CREATE INDEX IF NOT EXISTS idx_lc_from_to     ON land_changes (from_type, to_type);

-- ---------------------------------------------------------------------------
-- Table: state_summary
-- Materialized monthly rollup per state for fast dashboard queries.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS state_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    state           TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    total_area_ha   REAL    NOT NULL,
    total_co2_tons  REAL    NOT NULL,
    dominant_land   TEXT    NOT NULL,
    record_count    INTEGER NOT NULL,
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(state, year, month)
);

-- ---------------------------------------------------------------------------
-- Table: anomaly_flags
-- Records detected anomalies for downstream ArcGIS dashboard highlighting.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomaly_flags (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id       INTEGER NOT NULL REFERENCES land_use_records(id),
    state           TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    land_type       TEXT    NOT NULL,
    co2_tons        REAL    NOT NULL,
    z_score         REAL    NOT NULL,                        -- Standard deviations from mean
    flagged_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
