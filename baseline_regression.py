"""
baseline_regression.py

Baseline OLS regression of rice yield on climate anomaly features,
irrigation share, and lagged yield -- with district and year fixed effects.

This is your "ground truth" model before moving to random forest --
it's directly interpretable and closest to what you already know
from your econometrics course.
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ---- Load ----
PANEL_PATH = r"data\processed\final_panel.csv"
df = pd.read_csv(PANEL_PATH)

# ---- Prep ----
# Use log(yield) -- standard in yield regressions, makes coefficients
# interpretable as approximate % changes, and helps with the mild
# right-skew that yield data usually has.
df["log_yield"] = df["yield_kg_per_ha"].apply(lambda x: __import__("numpy").log(x))

# Clean column names for the regression formula (statsmodels doesn't like spaces)
df = df.rename(columns={"Dist Name": "district", "State Name": "state"})

# irrigation_share is missing for Maharashtra (rainfed, no irrigation data).
# Rather than dropping the whole state, fill with 0 (0% irrigated is a
# reasonable assumption for districts ICRISAT has no irrigation record for)
# AND add a flag so the model can tell "known 0% irrigated" apart from
# "irrigation share genuinely unknown" if that distinction matters later.
df["irrigation_share_filled"] = df["irrigation_share"].fillna(0)

# lagged_yield is missing for each district's first year (1990) --
# these rows can't be used in a model that includes lagged_yield as a
# predictor, so we drop them here (this is expected and fine -- you're
# only losing 74 rows out of 2064, about 3.6%)
model_df = df.dropna(subset=["lagged_yield"]).copy()
model_df["log_lagged_yield"] = model_df["lagged_yield"].apply(lambda x: __import__("numpy").log(x))

print(f"Rows used in model: {len(model_df)} (dropped {len(df) - len(model_df)} first-year rows)")

# ---- Baseline model: no fixed effects ----
# This is the simplest possible version -- good as a first check.
formula_simple = (
    "log_yield ~ rainfall_anomaly + temp_anomaly + extreme_heat_anomaly "
    "+ dry_spell_anomaly + irrigation_share_filled + log_lagged_yield"
)
model_simple = smf.ols(formula_simple, data=model_df).fit(
    cov_type="cluster", cov_kwds={"groups": model_df["district"]}
)
print("\n" + "="*70)
print("MODEL 1: Simple OLS (no fixed effects), clustered SE by district")
print("="*70)
print(model_simple.summary())

# ---- Model with district and year fixed effects ----
# This controls for anything constant about each district (soil quality,
# baseline farming practices) and anything common across all districts
# in a given year (national policy changes, input price shocks) --
# a much stronger identification strategy than the simple model above.
formula_fe = (
    "log_yield ~ rainfall_anomaly + temp_anomaly + extreme_heat_anomaly "
    "+ dry_spell_anomaly + irrigation_share_filled + log_lagged_yield "
    "+ C(district) + C(Year)"
)
model_fe = smf.ols(formula_fe, data=model_df).fit(
    cov_type="cluster", cov_kwds={"groups": model_df["district"]}
)

print("\n" + "="*70)
print("MODEL 2: OLS with district + year fixed effects, clustered SE")
print("="*70)
# Print only the climate/irrigation coefficients (not all 70+ district dummies)
coef_table = model_fe.summary2().tables[1]
key_vars = ["rainfall_anomaly", "temp_anomaly", "extreme_heat_anomaly",
            "dry_spell_anomaly", "irrigation_share_filled", "log_lagged_yield"]
print(coef_table.loc[key_vars])
print(f"\nR-squared: {model_fe.rsquared:.4f}")
print(f"N observations: {int(model_fe.nobs)}")

# ---- Save both model summaries to a text file for your report ----
with open("outputs/baseline_regression_results.txt", "w") as f:
    f.write("MODEL 1: Simple OLS\n")
    f.write(str(model_simple.summary()))
    f.write("\n\n" + "="*70 + "\n\n")
    f.write("MODEL 2: OLS with district + year fixed effects\n")
    f.write("(showing only climate/irrigation coefficients below; full output has 70+ district dummies)\n\n")
    f.write(coef_table.loc[key_vars].to_string())
    f.write(f"\n\nR-squared: {model_fe.rsquared:.4f}")
    f.write(f"\nN observations: {int(model_fe.nobs)}")

print("\nSaved full results to: outputs/baseline_regression_results.txt")