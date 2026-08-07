"""
Clean ICRISAT rice irrigated area data — same district/year filtering as yield panel.
Outputs clean rice irrigated area (1000 ha) per district-year.
Note: irrigation_share = rice_irrigated_area / rice_area_sown — needs separate rice area file.
"""

import pandas as pd
import numpy as np
import os

# ---- Settings ----
RAW_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\yield\raw\ICRISAT-District Level Data Irrigation.csv"
OUTPUT_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\rice_irrigated_area_clean.csv"
IRRIG_COL = "RICE IRRIGATED AREA (1000 ha)"
MIN_VALID_YEARS = 20  # same threshold as yield cleaning

# ---- Step 1: Load ----
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} rows, {df['Dist Name'].nunique()} districts, {df['State Name'].nunique()} states.")

# ---- Step 2: Mark missing/placeholder values ----
# Treat <=0 as missing (irrigated area can't be negative; 0 = not reported)
df["valid"] = df[IRRIG_COL] > 0
df["irrig_clean"] = df[IRRIG_COL].where(df["valid"])

n_missing = (~df["valid"]).sum()
print(f"Marked {n_missing} rows ({n_missing/len(df):.1%}) as missing (<=0 placeholder values).")

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

# ---- Step 4: Filter to kept districts ----
kept_keys = set(zip(kept["State Name"], kept["Dist Name"]))
df["_keep"] = df.apply(lambda r: (r["State Name"], r["Dist Name"]) in kept_keys, axis=1)
df_clean = df[df["_keep"]].drop(columns=["valid", "_keep"])

# Drop individual missing rows within kept districts
df_clean = df_clean.dropna(subset=["irrig_clean"])

print(f"\nFinal clean panel: {len(df_clean)} rows, "
      f"{df_clean['Dist Name'].nunique()} districts, "
      f"{df_clean['State Name'].nunique()} states, "
      f"years {df_clean['Year'].min()}-{df_clean['Year'].max()}")

# ---- Step 5: Save ----
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned irrigated area to: {OUTPUT_PATH}")

# ---- Step 6: Show coverage comparison with yield ----
print("\n=== COVERAGE PER STATE ===")
cov = df_clean.groupby('State Name').agg(
    districts=('Dist Name', 'nunique'),
    years=('Year', 'nunique'),
    rows=('Year', 'count')
).sort_values('districts', ascending=False)
print(cov.to_string())

# ---- Note on irrigation_share ----
print("""
NOTE: To compute irrigation_share = rice_irrigated_area / rice_area_sown,
you need the RICE AREA (1000 ha) file from ICRISAT (separate download).
If you have that file, place it at:
  yield/raw/ICRISAT-District Level Data Rice Area.csv
and I'll give you a merge snippet.
""")