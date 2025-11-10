"""
viz/dashboard_export.py
-----------------------
Multi-State Land Use Emissions Analysis
Generates three publication-ready dashboard chart PNGs:
  (a) outputs/emissions_by_state.png        — Bar chart: net CO₂ by state
  (b) outputs/land_cover_timeseries.png     — Line chart: area by land type over time
  (c) outputs/co2_forecast_vs_actual.png    — Scatter: forecast vs. actual CO₂

Loads data from SQLite DB if available, otherwise generates standalone synthetic data.

Run:  python viz/dashboard_export.py

Author: Deshraj Jogiya
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Paths & Logging
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "outputs" / "land_use_emissions.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] dashboard_export.py - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DARK_BG     = "#0F0F1A"
PANEL_BG    = "#16213E"
GRID_COLOR  = "#1E2A4A"
TEXT_COLOR  = "#E0E0F0"
ACCENT      = "#4C9BE8"

STATE_PALETTE = {
    "AL": "#4C9BE8",
    "CA": "#FF6B6B",
    "TX": "#F5A623",
    "WA": "#6CC5A4",
    "FL": "#9B59B6",
}

LAND_TYPE_PALETTE = {
    "forest":      "#2ECC71",
    "urban":       "#E74C3C",
    "agriculture": "#F39C12",
    "wetland":     "#3498DB",
    "shrubland":   "#95A5A6",
}

STATES     = ["AL", "CA", "TX", "WA", "FL"]
LAND_TYPES = ["forest", "urban", "agriculture", "wetland", "shrubland"]
CO2_FACTORS = {
    "forest":      -3.5,
    "urban":       +2.1,
    "agriculture": +0.8,
    "wetland":     -1.2,
    "shrubland":   -0.3,
}
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------

def load_data_from_db() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM land_use_records")).scalar()
        if count == 0:
            return pd.DataFrame()
        df = pd.read_sql("SELECT * FROM land_use_records", con=engine)
        logger.info("Loaded %d records from DB for visualization.", len(df))
        return df
    except Exception as exc:
        logger.warning("DB load failed: %s", exc)
        return pd.DataFrame()


def generate_synthetic_data(n_days: int = 213) -> pd.DataFrame:
    """Creates a synthetic land_use_records-style DataFrame."""
    rng   = np.random.default_rng(RNG_SEED)
    dates = pd.date_range("2025-04-01", periods=n_days, freq="D")

    STATE_BASELINES = {
        "AL": {"forest": 9_500_000, "urban":   620_000, "agriculture": 3_200_000, "wetland":  400_000, "shrubland":  180_000},
        "CA": {"forest": 6_800_000, "urban": 2_400_000, "agriculture": 9_600_000, "wetland":  150_000, "shrubland": 4_200_000},
        "TX": {"forest": 5_600_000, "urban": 2_900_000, "agriculture":19_000_000, "wetland":  800_000, "shrubland": 5_300_000},
        "WA": {"forest":10_400_000, "urban":   720_000, "agriculture": 4_200_000, "wetland":  320_000, "shrubland": 1_100_000},
        "FL": {"forest": 6_100_000, "urban": 1_800_000, "agriculture": 4_400_000, "wetland": 2_900_000, "shrubland":  400_000},
    }

    rows = []
    for state in STATES:
        for land_type in LAND_TYPES:
            base = STATE_BASELINES[state][land_type]
            for d in dates:
                doy  = d.day_of_year
                s    = 1 + 0.01 * np.sin(2 * np.pi * doy / 365)
                noise = rng.normal(0, 0.005)
                area  = max(0, base * s * (1 + noise))
                co2   = area * CO2_FACTORS[land_type]
                rows.append({
                    "state": state, "date": d.strftime("%Y-%m-%d"),
                    "land_type": land_type, "area_ha": area, "co2_tons": co2,
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Generated %d synthetic rows for visualization.", len(df))
    return df


# ---------------------------------------------------------------------------
# Chart A: Net CO₂ Emissions by State (Bar Chart)
# ---------------------------------------------------------------------------

def chart_emissions_by_state(df: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / "emissions_by_state.png"

    state_co2 = (
        df.groupby("state")["co2_tons"]
          .sum()
          .reindex(STATES)
          .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    x = np.arange(len(STATES))
    bar_colors = [STATE_PALETTE[s] for s in STATES]

    bars = ax.bar(x, state_co2.values / 1e6, color=bar_colors,
                  width=0.55, edgecolor="none", zorder=3)

    # Annotate bars
    for bar, val in zip(bars, state_co2.values / 1e6):
        sign   = "▲" if val > 0 else "▼"
        ypos   = bar.get_height() + (0.05 if val >= 0 else -0.15)
        color  = "#FF6B6B" if val > 0 else "#6CC5A4"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{sign} {abs(val):.2f}M", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color, zorder=5)

    # Reference line at 0
    ax.axhline(0, color="#FFFFFF", linewidth=0.8, alpha=0.4, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(STATES, color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_xlabel("U.S. State", color=TEXT_COLOR, fontsize=11, labelpad=8)
    ax.set_ylabel("Net CO₂ (millions of tons)", color=TEXT_COLOR, fontsize=11, labelpad=8)
    ax.set_title(
        "Net CO₂ Emissions by State\nApril – October 2025",
        color="white", fontsize=15, fontweight="bold", pad=16,
    )

    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}M"))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)

    # Legend: emitter vs sink
    emitter = mpatches.Patch(color="#FF6B6B", label="Net Emitter (positive CO₂)")
    sink    = mpatches.Patch(color="#6CC5A4", label="Net Carbon Sink (negative CO₂)")
    ax.legend(handles=[emitter, sink], loc="upper right",
              facecolor=PANEL_BG, edgecolor="none",
              labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info("Chart A saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Chart B: Land Cover Area Change Over Time (Line Chart)
# ---------------------------------------------------------------------------

def chart_land_cover_timeseries(df: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / "land_cover_timeseries.png"

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Aggregate: weekly average area per land type (across all states)
    df_agg = (
        df.groupby([pd.Grouper(key="date", freq="W"), "land_type"])["area_ha"]
          .sum()
          .reset_index()
    )
    df_pivot = df_agg.pivot(index="date", columns="land_type", values="area_ha").fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for land_type in LAND_TYPES:
        if land_type not in df_pivot.columns:
            continue
        series = df_pivot[land_type] / 1e6  # convert to millions ha
        color  = LAND_TYPE_PALETTE[land_type]
        ax.plot(df_pivot.index, series, label=land_type.capitalize(),
                color=color, linewidth=2.0, alpha=0.9, zorder=3)
        ax.fill_between(df_pivot.index, series, alpha=0.07, color=color)

    ax.set_xlabel("Date", color=TEXT_COLOR, fontsize=11, labelpad=8)
    ax.set_ylabel("Total Area (millions of hectares)", color=TEXT_COLOR, fontsize=11, labelpad=8)
    ax.set_title(
        "Land Cover Area Change Over Time\nAll 5 States Combined  |  Apr – Oct 2025",
        color="white", fontsize=15, fontweight="bold", pad=16,
    )
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}M ha"))
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper left", facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                        labelcolor=TEXT_COLOR, fontsize=9, ncol=3, framealpha=0.9)
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info("Chart B saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Chart C: CO₂ Forecast vs. Actual (Scatter Plot)
# ---------------------------------------------------------------------------

def chart_forecast_vs_actual(df: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / "co2_forecast_vs_actual.png"

    # Build feature matrix
    df = df.copy()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        df["day_of_year"] = dates.dt.day_of_year.fillna(1).astype(int)
        df["month"]       = dates.dt.month.fillna(1).astype(int)

    state_map = {s: i for i, s in enumerate(sorted(STATES))}
    lt_map    = {lt: i for i, lt in enumerate(sorted(LAND_TYPES))}
    df["state_enc"] = df["state"].map(state_map).fillna(0)
    df["lt_enc"]    = df["land_type"].map(lt_map).fillna(0)

    feature_cols = ["day_of_year", "month", "area_ha", "lt_enc", "state_enc"]
    df = df.dropna(subset=feature_cols + ["co2_tons"])

    sample = df.sample(min(5000, len(df)), random_state=RNG_SEED)
    X = sample[feature_cols].values
    y = sample["co2_tons"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RNG_SEED)

    # Train two models, pick Random Forest for scatter (better fit)
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RNG_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)

    # Scale to millions for readability
    scale = 1e6
    y_te_s    = y_te    / scale
    y_pred_lr_s = y_pred_lr / scale
    y_pred_rf_s = y_pred_rf / scale

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(DARK_BG)
    titles  = ["Linear Regression", "Random Forest"]
    preds   = [y_pred_lr_s, y_pred_rf_s]
    colors  = [ACCENT, "#6CC5A4"]

    for ax, title, pred, color in zip(axes, titles, preds, colors):
        ax.set_facecolor(PANEL_BG)

        # Hexbin for density at scale
        hb = ax.hexbin(y_te_s, pred, gridsize=40,
                       cmap="Blues" if color == ACCENT else "Greens",
                       mincnt=1, linewidths=0.1, alpha=0.85, zorder=3)
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        cb.set_label("Count", color=TEXT_COLOR, fontsize=8)

        # Perfect prediction line
        lims = [min(y_te_s.min(), pred.min()), max(y_te_s.max(), pred.max())]
        ax.plot(lims, lims, "--", color="#FFFFFF", linewidth=1.2, alpha=0.5, zorder=4, label="Perfect fit")

        from sklearn.metrics import r2_score as r2s
        r2_val = r2s(y_te_s, pred)
        ax.set_title(f"{title}\nR² = {r2_val:.4f}",
                     color="white", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Actual CO₂ (M tons)", color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, zorder=1)

    axes[0].set_ylabel("Predicted CO₂ (M tons)", color=TEXT_COLOR, fontsize=10)

    fig.suptitle(
        "CO₂ Forecast vs. Actual  |  Land Emissions Models",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info("Chart C saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_dashboard() -> None:
    logger.info("=" * 65)
    logger.info("Dashboard Export — Generating 3 Charts")
    logger.info("=" * 65)

    df = load_data_from_db()
    if df.empty:
        logger.info("DB not available — using synthetic data.")
        df = generate_synthetic_data()

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    paths = []

    logger.info("Generating Chart A: Emissions by State ...")
    paths.append(chart_emissions_by_state(df))

    logger.info("Generating Chart B: Land Cover Timeseries ...")
    paths.append(chart_land_cover_timeseries(df))

    logger.info("Generating Chart C: CO₂ Forecast vs Actual ...")
    paths.append(chart_forecast_vs_actual(df))

    print("\n--- Dashboard Export Complete ---")
    for p in paths:
        print(f"  [OK] {p}")
    print()


if __name__ == "__main__":
    generate_dashboard()
