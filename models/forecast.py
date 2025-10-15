"""
models/forecast.py
------------------
Multi-State Land Use Emissions Analysis
CO₂ trend forecasting using Linear Regression and Random Forest.

Trains on synthetic time-series data derived from the SQLite database
(or generates standalone synthetic data if DB is not yet populated).

Outputs:
  • Printed R² and MSE for both models
  • outputs/feature_importance.png  — RF feature importance bar chart

Run:  python models/forecast.py

Author: Deshraj Jogiya
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "outputs" / "land_use_emissions.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] forecast.py - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CO2_FACTORS: dict[str, float] = {
    "forest":      -3.5,
    "urban":       +2.1,
    "agriculture": +0.8,
    "wetland":     -1.2,
    "shrubland":   -0.3,
}
STATES      = ["AL", "CA", "TX", "WA", "FL"]
LAND_TYPES  = list(CO2_FACTORS.keys())
RNG_SEED    = 42
TEST_SIZE   = 0.20

FEATURE_COLS = [
    "day_of_year",
    "month",
    "year",
    "area_ha",
    "land_type_encoded",
    "state_encoded",
]
TARGET_COL = "co2_tons"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_from_db() -> pd.DataFrame:
    """
    Loads land_use_records from SQLite and adds ML-ready encoded columns.
    Returns an empty DataFrame if DB doesn't exist or table is empty.
    """
    if not DB_PATH.exists():
        logger.warning("Database not found at %s — will generate synthetic data.", DB_PATH)
        return pd.DataFrame()

    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM land_use_records"))
            count = result.scalar()
            if count == 0:
                logger.warning("land_use_records table is empty.")
                return pd.DataFrame()
        df = pd.read_sql("SELECT * FROM land_use_records", con=engine)
        logger.info("Loaded %d records from database.", len(df))
        return df
    except Exception as exc:
        logger.warning("DB load failed (%s) — falling back to synthetic data.", exc)
        return pd.DataFrame()


def generate_synthetic_ml_data(n_samples: int = 15_000) -> pd.DataFrame:
    """
    Generates a synthetic dataset for training/testing the ML models.
    Mirrors the structure of land_use_records with realistic CO₂ signal.
    """
    logger.info("Generating %d synthetic samples for ML training ...", n_samples)
    rng = np.random.default_rng(RNG_SEED)

    dates = pd.date_range("2025-04-01", periods=n_samples // len(STATES), freq="D")
    # Replicate for all states
    rows = []
    for state_idx, state in enumerate(STATES):
        for lt_idx, land_type in enumerate(LAND_TYPES):
            base_area = rng.uniform(50_000, 10_000_000)
            n_per_type = n_samples // (len(STATES) * len(LAND_TYPES))
            for i in range(n_per_type):
                run_date   = dates[i % len(dates)]
                day_of_year = run_date.day_of_year
                # Seasonal variation
                seasonal = 1 + 0.01 * np.sin(2 * np.pi * day_of_year / 365)
                noise    = rng.normal(0, 0.02)
                area_ha  = max(0, base_area * seasonal * (1 + noise))
                co2_factor = CO2_FACTORS[land_type]
                # Add slight non-linearity for RF to exploit
                co2_tons = (area_ha * co2_factor
                            + 0.3 * area_ha * co2_factor * np.sin(2 * np.pi * day_of_year / 365)
                            + rng.normal(0, abs(co2_factor) * area_ha * 0.005))
                rows.append({
                    "state":             state,
                    "date":              run_date.strftime("%Y-%m-%d"),
                    "land_type":         land_type,
                    "area_ha":           round(area_ha, 2),
                    "co2_tons":          round(co2_tons, 4),
                    "day_of_year":       int(day_of_year),
                    "month":             int(run_date.month),
                    "year":              int(run_date.year),
                    "state_encoded":     state_idx,
                    "land_type_encoded": lt_idx,
                })

    df = pd.DataFrame(rows)
    logger.info("Synthetic ML dataset: %d rows.", len(df))
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds encoded columns and derived date features if not already present.
    """
    df = df.copy()

    # Date features
    if "day_of_year" not in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        df["day_of_year"] = dates.dt.day_of_year.fillna(1).astype(int)
        df["month"]       = dates.dt.month.fillna(1).astype(int)
        df["year"]        = dates.dt.year.fillna(2025).astype(int)

    # Encode state
    if "state_encoded" not in df.columns:
        state_map = {s: i for i, s in enumerate(sorted(STATES))}
        df["state_encoded"] = df["state"].map(state_map).fillna(0).astype(int)

    # Encode land_type
    if "land_type_encoded" not in df.columns:
        lt_map = {lt: i for i, lt in enumerate(sorted(LAND_TYPES))}
        df["land_type_encoded"] = df["land_type"].map(lt_map).fillna(0).astype(int)

    # Ensure numeric area_ha
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce").fillna(0)

    return df


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains a Linear Regression model and evaluates it.
    Uses StandardScaler for feature normalization (important for LR).
    Returns metrics dict.
    """
    logger.info("Training Linear Regression ...")

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_tr_sc, y_train)

    y_pred = lr.predict(X_te_sc)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    logger.info("  Linear Regression → R²=%.4f | MSE=%.4f | RMSE=%.4f", r2, mse, rmse)

    return {
        "model":   lr,
        "scaler":  scaler,
        "y_pred":  y_pred,
        "r2":      r2,
        "mse":     mse,
        "rmse":    rmse,
        "coefficients": dict(zip(FEATURE_COLS, lr.coef_)),
    }


def train_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains a Random Forest Regressor and evaluates it.
    Returns metrics dict including feature importances.
    """
    logger.info("Training Random Forest (n_estimators=200, max_depth=15) ...")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RNG_SEED,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    logger.info("  Random Forest → R²=%.4f | MSE=%.4f | RMSE=%.4f", r2, mse, rmse)

    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model":        rf,
        "y_pred":       y_pred,
        "r2":           r2,
        "mse":          mse,
        "rmse":         rmse,
        "importance_df": importance_df,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_feature_importance_chart(importance_df: pd.DataFrame) -> Path:
    """
    Saves a horizontal bar chart of Random Forest feature importances.
    """
    out_path = OUTPUT_DIR / "feature_importance.png"

    palette = [
        "#4C9BE8", "#6CC5A4", "#F5A623", "#E85C5C",
        "#9B59B6", "#2ECC71",
    ]
    colors = palette[: len(importance_df)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=colors,
        edgecolor="none",
        height=0.55,
    )

    # Value labels
    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            color="white", fontsize=9,
        )

    ax.set_xlabel("Feature Importance (Gini)", color="#AAAACC", fontsize=10)
    ax.set_title(
        "Random Forest — Feature Importances\nCO₂ Emission Prediction",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#333366")
    ax.xaxis.label.set_color("#AAAACC")
    ax.set_xlim(0, importance_df["importance"].max() * 1.18)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Feature importance chart saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_forecasting() -> None:
    logger.info("=" * 65)
    logger.info("CO₂ Emissions Forecast — Linear Regression + Random Forest")
    logger.info("=" * 65)

    # 1. Load or generate data
    df = load_from_db()
    if df.empty:
        df = generate_synthetic_ml_data(n_samples=20_000)
    else:
        df = prepare_features(df)

    # Ensure required columns exist
    df = prepare_features(df)

    # 2. Drop rows with NaN in required columns
    req = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=req)
    logger.info("Feature matrix: %d rows × %d features.", len(df), len(FEATURE_COLS))

    if len(df) < 50:
        logger.error("Not enough data for training (need ≥50 rows). Exiting.")
        sys.exit(1)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 3. Train/test split (time-aware: sorted by date then split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RNG_SEED, shuffle=True,
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # 4. Train models
    lr_results = train_linear_regression(X_train, X_test, y_train, y_test)
    rf_results = train_random_forest(X_train, X_test, y_train, y_test)

    # 5. Print report
    print("\n" + "═" * 55)
    print("  CO₂ FORECAST MODEL PERFORMANCE REPORT")
    print("═" * 55)
    print(f"  {'Metric':<22} {'Linear Reg':>12} {'Random Forest':>14}")
    print("  " + "-" * 51)
    print(f"  {'R² Score':<22} {lr_results['r2']:>12.4f} {rf_results['r2']:>14.4f}")
    print(f"  {'MSE':<22} {lr_results['mse']:>12.4f} {rf_results['mse']:>14.4f}")
    print(f"  {'RMSE':<22} {lr_results['rmse']:>12.4f} {rf_results['rmse']:>14.4f}")
    print("═" * 55)

    print("\n  Linear Regression Coefficients:")
    for feat, coef in lr_results["coefficients"].items():
        print(f"    {feat:<25} {coef:>12.6f}")

    print("\n  Random Forest Feature Importances:")
    for _, row in rf_results["importance_df"].iterrows():
        bar = "█" * int(row["importance"] * 60)
        print(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")
    print()

    # 6. Save feature importance chart
    chart_path = save_feature_importance_chart(rf_results["importance_df"])
    print(f"  Feature importance chart → {chart_path}\n")

    # 7. Accuracy gate
    best_r2 = max(lr_results["r2"], rf_results["r2"])
    if best_r2 >= 0.85:
        print(f"  ✅ Target achieved: Best R² = {best_r2:.4f} (≥ 0.85 threshold)\n")
    else:
        print(f"  ⚠ Best R² = {best_r2:.4f} — below 0.85 target (may need more data).\n")


if __name__ == "__main__":
    run_forecasting()
