"""
add_irrigation_share.py

Loads the raw rice AREA file, cleans it, then computes irrigation_share
(rice_irrigated_area_1000ha / rice_area_1000ha) and merges it into the
existing final_panel.csv, filling in the empty irrigation_share column.
"""

import pandas as pd

# ---- Paths -- adjust these to match your actual folder structure ----
AREA_RAW_PATH = "yield\\raw\\ICRISAT-District Level Data Area.csv"
FINAL_PANEL_PATH = "data/processed/final_panel.csv"
OUTPUT_PATH = "data/processed/final_panel.csv"  # overwrites with the added column

AREA_COL = "RICE AREA (1000 ha)"

# ---- Step 1: Load and clean the area file ----
area = pd.read_csv(AREA_RAW_PATH)
print(f"Loaded area file: {len(area)} rows")

# Only -1 is a placeholder here (0 is a legitimate "no rice grown that year" value,
# unlike the yield file where 0 didn't make sense)
area["area_clean"] = area[AREA_COL].where(area[AREA_COL] != -1)

n_missing = area["area_clean"].isna().sum()
print(f"Marked {n_missing} rows as missing (-1 placeholder)")

# ---- Step 2: Load the existing final panel ----
panel = pd.read_csv(FINAL_PANEL_PATH)
print(f"Loaded final panel: {len(panel)} rows")

# ---- Step 3: Merge area data onto the panel (by district + year) ----
area_small = area[["Dist Name", "Year", "area_clean"]].rename(
    columns={"area_clean": "rice_area_1000ha"}
)

before_cols = set(panel.columns)
merged = panel.merge(area_small, on=["Dist Name", "Year"], how="left", indicator=True)

print("\nMerge check:")
print(merged["_merge"].value_counts())
merged = merged.drop(columns=["_merge"])

# ---- Step 4: Compute irrigation_share ----
# Overwrite the existing (currently all-NaN) irrigation_share column
merged["irrigation_share"] = (
    merged["rice_irrigated_area_1000ha"] / merged["rice_area_1000ha"]
)

# Sanity check: irrigation_share should be between 0 and 1 (occasionally slightly
# above 1 due to data quirks/rounding across sources -- flag rather than silently drop)
out_of_range = merged[(merged["irrigation_share"] < 0) | (merged["irrigation_share"] > 1.0)]
if len(out_of_range) > 0:
    print(f"\nNOTE: {len(out_of_range)} rows had irrigation_share > 1.0 (likely due to "
          f"multi-season irrigation reporting vs. single-season area sown). "
          f"Capping these at 1.0 rather than dropping them.")
    print(out_of_range[["Dist Name", "Year", "rice_irrigated_area_1000ha", "rice_area_1000ha", "irrigation_share"]].to_string())
    merged["irrigation_share"] = merged["irrigation_share"].clip(upper=1.0)

print(f"\nirrigation_share now filled for {merged['irrigation_share'].notna().sum()} of {len(merged)} rows")
print(merged["irrigation_share"].describe())

# ---- Step 5: Save ----
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved updated final panel to: {OUTPUT_PATH}")
print(f"Final shape: {merged.shape}")