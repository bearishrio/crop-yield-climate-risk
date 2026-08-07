"""
clean_icrisat_rice.py

Cleans the raw ICRISAT district-level rice yield export for
Maharashtra, Orissa, and Punjab (1990-2019).

What it does:
1. Loads the raw CSV
2. Treats -1 as a missing-data code (district didn't exist yet / not reported)
   and treats exactly 0 the same way (negligible/no data, not a real zero yield)
3. Counts how many valid (non-missing) years each district actually has
4. Keeps only districts with at least MIN_VALID_YEARS out of the full window
5. Saves a clean panel + prints a summary so you can see exactly what was dropped
"""

import pandas as pd

# ---- Settings you can adjust ----
RAW_PATH = "yield/raw/ICRISAT-District Level Data.csv"
OUTPUT_PATH = "data/processed/rice_yield_clean.csv"
YIELD_COL = "RICE YIELD (Kg per ha)"
MIN_VALID_YEARS = 20  # out of 30 (1990-2019) -- adjust if you want a stricter/looser bar

# ---- Step 1: Load ----
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} rows, {df['Dist Name'].nunique()} districts, "
      f"{df['State Name'].nunique()} states.")

# ---- Step 2: Mark missing/placeholder values ----
# -1 = district didn't exist yet / not reported that year
# 0  = treated as missing here too (negligible/no data, not a real observed zero yield)
df["valid"] = df[YIELD_COL] > 0
df["yield_clean"] = df[YIELD_COL].where(df["valid"])  # NaN where invalid

n_missing = (~df["valid"]).sum()
print(f"Marked {n_missing} rows ({n_missing/len(df):.1%}) as missing "
      f"(-1 or 0 placeholder values).")

# ---- Step 3: Count valid years per district ----
coverage = (
    df.groupby(["State Name", "Dist Name"])["valid"]
    .sum()
    .reset_index()
    .rename(columns={"valid": "valid_years"})
)

kept = coverage[coverage["valid_years"] >= MIN_VALID_YEARS]
dropped = coverage[coverage["valid_years"] < MIN_VALID_YEARS]

print(f"\nThreshold: keep districts with >= {MIN_VALID_YEARS} valid years")
print(f"Kept: {len(kept)} districts")
print(f"Dropped: {len(dropped)} districts")
if len(dropped) > 0:
    print("\nDropped districts:")
    print(dropped.sort_values("valid_years").to_string(index=False))

# ---- Step 4: Filter the main dataframe to kept districts only ----
kept_keys = set(zip(kept["State Name"], kept["Dist Name"]))
df["_keep"] = df.apply(lambda r: (r["State Name"], r["Dist Name"]) in kept_keys, axis=1)
df_clean = df[df["_keep"]].drop(columns=["valid", "_keep"])

# Optional: also drop the individual missing rows within kept districts
# (rather than keeping NaNs) -- comment this out if you'd rather keep and
# handle NaNs later during feature engineering / modeling.
df_clean = df_clean.dropna(subset=["yield_clean"])

print(f"\nFinal clean panel: {len(df_clean)} rows, "
      f"{df_clean['Dist Name'].nunique()} districts, "
      f"{df_clean['State Name'].nunique()} states, "
      f"years {df_clean['Year'].min()}-{df_clean['Year'].max()}")

# ---- Step 5: Save ----
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned panel to: {OUTPUT_PATH}")