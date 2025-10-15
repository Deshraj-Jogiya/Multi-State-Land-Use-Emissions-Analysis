-- =============================================================================
-- Multi-State Land Use Emissions Analysis
-- Analytical SQL Queries
-- Author: Deshraj Jogiya
-- Run against: outputs/land_use_emissions.db (SQLite)
-- =============================================================================


-- ---------------------------------------------------------------------------
-- QUERY 1: Total Net CO2 Emissions by State (All Time)
-- Purpose: Identify which states are net emitters vs. carbon sinks.
-- ---------------------------------------------------------------------------
SELECT
    state,
    ROUND(SUM(co2_tons), 2)                          AS total_co2_tons,
    ROUND(SUM(area_ha),  2)                          AS total_area_ha,
    ROUND(SUM(co2_tons) / NULLIF(SUM(area_ha), 0), 4) AS co2_per_ha,
    COUNT(*)                                          AS record_count,
    CASE
        WHEN SUM(co2_tons) < 0 THEN 'Net Carbon Sink'
        ELSE 'Net Emitter'
    END                                               AS carbon_status
FROM land_use_records
GROUP BY state
ORDER BY total_co2_tons DESC;


-- ---------------------------------------------------------------------------
-- QUERY 2: Top 10 Land Cover Transitions (Urban Expansion / Forest Loss)
-- Purpose: Track the largest land-use change events across all states.
-- ---------------------------------------------------------------------------
SELECT
    state,
    from_type,
    to_type,
    ROUND(SUM(area_changed), 2)  AS total_area_changed_ha,
    ROUND(SUM(co2_impact),   2)  AS total_co2_impact,
    COUNT(*)                     AS num_events,
    MIN(change_date)             AS earliest_change,
    MAX(change_date)             AS latest_change
FROM land_changes
GROUP BY state, from_type, to_type
ORDER BY total_area_changed_ha DESC
LIMIT 10;


-- ---------------------------------------------------------------------------
-- QUERY 3: Year-over-Year CO2 Emission Trends by State and Land Type
-- Purpose: Measure emissions growth/decline across years for forecasting validation.
-- ---------------------------------------------------------------------------
SELECT
    state,
    land_type,
    strftime('%Y', date)                                  AS year,
    ROUND(SUM(co2_tons), 2)                               AS annual_co2_tons,
    ROUND(SUM(area_ha),  2)                               AS annual_area_ha,
    COUNT(*)                                              AS observation_count
FROM land_use_records
GROUP BY state, land_type, year
ORDER BY state, land_type, year;


-- ---------------------------------------------------------------------------
-- QUERY 4: Anomaly Detection — Records Exceeding 2 Standard Deviations
-- Purpose: Flag unusual CO2 readings per land type for ArcGIS dashboard alerts.
-- ---------------------------------------------------------------------------
WITH stats AS (
    SELECT
        land_type,
        AVG(co2_tons)                                     AS mean_co2,
        -- SQLite lacks STDDEV; approximate via variance
        SQRT(AVG(co2_tons * co2_tons) - AVG(co2_tons) * AVG(co2_tons)) AS std_co2
    FROM land_use_records
    GROUP BY land_type
)
SELECT
    r.id,
    r.state,
    r.date,
    r.land_type,
    ROUND(r.co2_tons, 4)                                  AS co2_tons,
    ROUND(s.mean_co2, 4)                                  AS mean_co2,
    ROUND(s.std_co2,  4)                                  AS std_co2,
    ROUND(ABS(r.co2_tons - s.mean_co2) / NULLIF(s.std_co2, 0), 3) AS z_score,
    r.source_file
FROM land_use_records r
JOIN stats s ON r.land_type = s.land_type
WHERE ABS(r.co2_tons - s.mean_co2) > 2 * s.std_co2
ORDER BY z_score DESC
LIMIT 50;


-- ---------------------------------------------------------------------------
-- QUERY 5: Monthly Emission Summary with Rolling Context
-- Purpose: Power the monthly trend panel in the ArcGIS dashboard.
-- ---------------------------------------------------------------------------
SELECT
    state,
    strftime('%Y-%m', date)                               AS year_month,
    land_type,
    ROUND(SUM(co2_tons), 2)                               AS monthly_co2,
    ROUND(AVG(co2_tons), 4)                               AS avg_daily_co2,
    ROUND(SUM(area_ha),  2)                               AS monthly_area_ha,
    COUNT(*)                                              AS days_recorded
FROM land_use_records
GROUP BY state, year_month, land_type
ORDER BY state, year_month, land_type;


-- ---------------------------------------------------------------------------
-- QUERY 6: Forest Loss Leaders — States with Highest Forest-to-Other Conversion
-- Purpose: Identify deforestation hotspots for conservation prioritization.
-- ---------------------------------------------------------------------------
SELECT
    state,
    to_type                                               AS converted_to,
    ROUND(SUM(area_changed), 2)                           AS total_forest_lost_ha,
    ROUND(SUM(co2_impact),   2)                           AS total_co2_impact,
    COUNT(*)                                              AS conversion_events
FROM land_changes
WHERE from_type = 'forest'
GROUP BY state, to_type
ORDER BY total_forest_lost_ha DESC;


-- ---------------------------------------------------------------------------
-- QUERY 7: Daily Emission Rate — Peak Emission Days Per State
-- Purpose: Identify high-load emission days for temporal anomaly reporting.
-- ---------------------------------------------------------------------------
SELECT
    state,
    date,
    ROUND(SUM(co2_tons), 2)   AS daily_total_co2,
    ROUND(MAX(co2_tons), 2)   AS peak_co2_single_type,
    COUNT(DISTINCT land_type) AS land_types_recorded
FROM land_use_records
GROUP BY state, date
ORDER BY daily_total_co2 DESC
LIMIT 20;
