"""
visualize_ranking.py

Publication-style visualization of the district climate-sensitivity ranking.
The compound-scenario panel is included only when the ranking CSV contains
usable compound predictions.
"""

from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.patches import Patch

# Use a non-interactive backend so terminal runs do not require Tk/display
# support. The script saves files rather than opening a GUI window.
mpl.use("Agg")
import matplotlib.pyplot as plt


# Publication-style settings, matching visualize_results.py.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


STATE_COLORS = {
    "Punjab": "#1b1b1b",
    "Maharashtra": "#b35806",
    "Odisha": "#3a7ca5",
}
STATE_ORDER = ["Punjab", "Maharashtra", "Odisha"]
UNKNOWN_STATE_COLOR = "#777777"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKING_PATH = PROJECT_ROOT / "data" / "processed" / (
    "district_climate_sensitivity_ranking.csv"
)
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PNG_PATH = FIGURE_DIR / "climate_sensitivity_ranking.png"
PDF_PATH = FIGURE_DIR / "climate_sensitivity_ranking.pdf"


def parse_boolean(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to real booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def main() -> None:
    if not RANKING_PATH.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {RANKING_PATH}")

    ranking = pd.read_csv(RANKING_PATH)

    required_columns = {
        "district",
        "state",
        "irrigation_share",
        "predicted_yield_loss_pct",
    }
    missing_required = required_columns - set(ranking.columns)
    if missing_required:
        raise ValueError(
            "Ranking CSV is missing required columns: "
            + ", ".join(sorted(missing_required))
        )

    # Use Odisha in figures even if the source data uses the older label Orissa.
    ranking["state"] = ranking["state"].replace({"Orissa": "Odisha"})
    ranking["predicted_yield_loss_pct"] = pd.to_numeric(
        ranking["predicted_yield_loss_pct"], errors="coerce"
    )
    ranking["irrigation_share"] = pd.to_numeric(
        ranking["irrigation_share"], errors="coerce"
    )
    ranking = ranking.dropna(
        subset=["district", "state", "predicted_yield_loss_pct"]
    ).copy()

    if ranking.empty:
        raise ValueError("The ranking CSV has no usable rows for plotting.")

    has_compound = "predicted_yield_loss_compound_pct" in ranking.columns
    if has_compound:
        ranking["predicted_yield_loss_compound_pct"] = pd.to_numeric(
            ranking["predicted_yield_loss_compound_pct"], errors="coerce"
        )
        has_compound = ranking["predicted_yield_loss_compound_pct"].notna().any()

    has_reliability_flag = "low_performing_cv_fold" in ranking.columns
    if has_reliability_flag:
        ranking["low_performing_cv_fold"] = parse_boolean(
            ranking["low_performing_cv_fold"]
        )

    # Select and order the 15 highest rainfall-only sensitivity districts.
    top15 = (
        ranking.nlargest(15, "predicted_yield_loss_pct")
        .sort_values("predicted_yield_loss_pct", ascending=True)
    )

    n_panels = 2 if has_compound else 1
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(12 if n_panels == 2 else 7.5, 7),
        squeeze=False,
    )
    axes = axes[0]

    # Panel (a): top 15 rainfall-only sensitivity districts.
    ax = axes[0]
    for position, (_, row) in enumerate(top15.iterrows()):
        state = row["state"]
        color = STATE_COLORS.get(state, UNKNOWN_STATE_COLOR)
        hatch = "///" if (
            has_reliability_flag and row["low_performing_cv_fold"]
        ) else ""
        ax.barh(
            position,
            row["predicted_yield_loss_pct"],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            height=0.68,
            hatch=hatch,
        )

    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["district"])
    ax.set_title("(a) Top 15 districts: rainfall-only shock")
    ax.set_xlabel("Predicted yield loss (%, -1 SD rainfall shock)")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)

    state_handles = [
        Patch(
            facecolor=STATE_COLORS[state],
            edgecolor="black",
            linewidth=0.5,
            label=state,
        )
        for state in STATE_ORDER
        if state in set(top15["state"])
    ]
    if has_reliability_flag and top15["low_performing_cv_fold"].any():
        state_handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label="Low-performing CV fold",
            )
        )
    ax.legend(handles=state_handles, frameon=False, loc="lower right")

    # Panel (b): state averages, only when compound values are available.
    if has_compound:
        ax = axes[1]
        state_means = (
            ranking.groupby("state", as_index=False)
            .agg(
                rainfall_only=("predicted_yield_loss_pct", "mean"),
                compound=("predicted_yield_loss_compound_pct", "mean"),
            )
        )
        state_means["state_order"] = state_means["state"].map(
            {state: i for i, state in enumerate(STATE_ORDER)}
        )
        state_means = state_means.sort_values("state_order")

        x_positions = list(range(len(state_means)))
        bar_width = 0.36
        state_colors = [
            STATE_COLORS.get(state, UNKNOWN_STATE_COLOR)
            for state in state_means["state"]
        ]

        ax.bar(
            [x - bar_width / 2 for x in x_positions],
            state_means["rainfall_only"],
            width=bar_width,
            color=state_colors,
            edgecolor="black",
            linewidth=0.5,
            label="Rainfall-only",
        )
        ax.bar(
            [x + bar_width / 2 for x in x_positions],
            state_means["compound"],
            width=bar_width,
            color=state_colors,
            edgecolor="black",
            linewidth=0.5,
            hatch="//",
            alpha=0.75,
            label="Compound rainfall + temperature",
        )

        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(state_means["state"])
        ax.set_title("(b) Average sensitivity by state")
        ax.set_xlabel("State")
        ax.set_ylabel("Average predicted yield loss (%)")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(ranking)} district rows from: {RANKING_PATH}")
    print(f"Compound panel included: {has_compound}")
    if not has_compound:
        print(
            "Note: predicted_yield_loss_compound_pct was absent or entirely "
            "missing; only panel (a) was produced."
        )
    if not has_reliability_flag:
        print(
            "Note: low_performing_cv_fold was absent; reliability hatching "
            "was not added."
        )
    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
