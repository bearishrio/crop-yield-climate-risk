"""
visualize_results.py

Publication-style 4-panel figure summarizing the project's data and
key findings, styled to look like a figure from an economics/ag-econ
journal article rather than a colorful dashboard: serif fonts, muted
palette, panel labels (a)-(d), minimal gridlines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Publication-style settings ----
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

# Muted, print-friendly palette (colorblind-safe, works in grayscale too)
state_colors = {"Punjab": "#1b1b1b", "Maharashtra": "#b35806", "Odisha": "#3a7ca5"}
state_markers = {"Punjab": "o", "Maharashtra": "s", "Odisha": "^"}

# ---- Load data ----
panel = pd.read_csv("data/processed/final_panel.csv")
ranking = pd.read_csv("data/processed/district_climate_sensitivity_ranking.csv")

panel["State Name"] = panel["State Name"].replace({"Orissa": "Odisha"})
ranking["state"] = ranking["state"].replace({"Orissa": "Odisha"})

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# ---- Panel (a): State-average yield over time ----
ax = axes[0, 0]
yearly_state_yield = (
    panel.groupby(["Year", "State Name"])["yield_kg_per_ha"].mean().reset_index()
)
for state in ["Punjab", "Maharashtra", "Odisha"]:
    sub = yearly_state_yield[yearly_state_yield["State Name"] == state]
    ax.plot(sub["Year"], sub["yield_kg_per_ha"], label=state,
            color=state_colors[state], linewidth=1.3,
            marker=state_markers[state], markersize=3, markevery=3)
ax.set_title("(a) Mean rice yield by state, 1990\u20132019")
ax.set_xlabel("Year")
ax.set_ylabel("Yield (kg/ha)")
ax.legend(frameon=False, loc="upper left")
ax.grid(True, axis="y")
ax.grid(False, axis="x")

# ---- Panel (b): Rainfall anomaly distribution ----
ax = axes[0, 1]
bins = 30
for state in ["Punjab", "Maharashtra", "Odisha"]:
    sub = panel[panel["State Name"] == state]["rainfall_anomaly"]
    ax.hist(sub, bins=bins, histtype="step", linewidth=1.4,
            color=state_colors[state], label=state, density=True)
ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
ax.set_title("(b) Distribution of rainfall anomalies")
ax.set_xlabel("Rainfall anomaly (SD from district baseline)")
ax.set_ylabel("Density")
ax.legend(frameon=False, loc="upper left")
ax.grid(True, axis="y")
ax.grid(False, axis="x")

# ---- Panel (c): Irrigation share vs. sensitivity ----
ax = axes[1, 0]
for state in ["Punjab", "Maharashtra", "Odisha"]:
    sub = ranking[ranking["state"] == state]
    ax.scatter(sub["irrigation_share"], sub["predicted_yield_loss_pct"],
               label=state, color=state_colors[state],
               marker=state_markers[state], s=28,
               facecolors="none" if state != "Punjab" else state_colors[state],
               linewidth=1.0)
ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax.set_title("(c) Irrigation share and predicted climate sensitivity")
ax.set_xlabel("Irrigation share")
ax.set_ylabel("Predicted yield loss (%),\n\u22121 SD rainfall shock")
ax.legend(frameon=False, loc="upper right")
ax.grid(True)

# ---- Panel (d): Top 15 most sensitive districts ----
ax = axes[1, 1]
top15 = ranking.nlargest(15, "predicted_yield_loss_pct").sort_values("predicted_yield_loss_pct")
bar_colors = [state_colors[s] for s in top15["state"]]
ax.barh(top15["district"], top15["predicted_yield_loss_pct"],
        color=bar_colors, edgecolor="black", linewidth=0.5, height=0.65)
ax.set_title("(d) Fifteen most climate-sensitive districts")
ax.set_xlabel("Predicted yield loss (%), \u22121 SD rainfall shock")
ax.grid(True, axis="x")
ax.grid(False, axis="y")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()

# ---- Save (high-res, suitable for print/PDF embedding) ----
output_path = "outputs/figures/project_dashboard.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved dashboard to: {output_path}")

# Also save a PDF version -- vector graphics look cleaner in a LaTeX report
pdf_path = "outputs/figures/project_dashboard.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved vector version to: {pdf_path}")

plt.show()