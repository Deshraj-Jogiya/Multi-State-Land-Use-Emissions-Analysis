"""Chart of the year-by-year backtest: viz/preview/epa_backtest_by_year.png.

Accuracy (1 - WAPE) of each model for each target year 2015-2022 (exports/epa_backtest_by_year.csv). Last year's value is the one
line in the accent colour: it is the bar every model has to clear.

Usage: python viz/backtest_chart.py   (after models/epa_backtest.py)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viz" / "preview" / "epa_backtest_by_year.png"
BG, TEXT, MUTED, GRID, ACCENT = "#111315", "#ecebe6", "#a8acb2", "#2a2e33", "#8cc4ad"
OTHERS = {"Linear Regression": "#d9d8d2", "Random Forest": "#9ea3a9", "3-year average": "#6f747a", "Recent trend": "#4f5359"}
NAIVE = "Last year's value"

for f in (ROOT / "viz" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["IBM Plex Sans", "DejaVu Sans"], "text.color": TEXT, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def main() -> Path:
    by_year = pd.read_csv(ROOT / "exports" / "epa_backtest_by_year.csv")
    summary = pd.read_csv(ROOT / "exports" / "epa_backtest_summary.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    lines = {}
    for model in [*OTHERS, NAIVE]:
        g = by_year[by_year["model"] == model].sort_values("year")
        colour = ACCENT if model == NAIVE else OTHERS[model]
        lines[model] = ax.plot(g["year"], g["accuracy"] * 100, color=colour, lw=2.6 if model == NAIVE else 1.6, marker="o", ms=4, zorder=3 if model == NAIVE else 2)[0]
    lo = min(by_year["accuracy"].min() * 100 - 1, 89)
    ax.set_xlim(2014.5, 2022.5)
    ax.set_ylim(lo, 100)
    ax.set_xticks(range(2015, 2023))
    ax.set_ylabel("accuracy, 1 - WAPE (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    order = summary.sort_values("mean_accuracy", ascending=False).index
    leg = ax.legend([lines[m] for m in order], [f"{m}  {summary.loc[m, 'mean_accuracy'] * 100:.1f}%" for m in order],
                    title="mean over 8 years", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9.5,
                    title_fontsize=9, alignment="left")
    leg.get_title().set_color(MUTED)
    for text, m in zip(leg.get_texts(), order):
        text.set_color(ACCENT if m == NAIVE else TEXT)
    wins = int(summary.loc["Linear Regression", "years_beating_last_year"])
    fig.text(0.06, 0.95, "Repeating last year's value is hard to beat", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.905, f"EPA state land-use series, 5 states x 5 land types, forecast one year ahead from 8 origins. "
             f"Linear Regression beat it in {wins} of 8 years.", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.8, top=0.85, bottom=0.08)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print("wrote", main())
