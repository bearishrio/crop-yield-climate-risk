"""
Merge cleaned irrigation data into final_panel.csv.
Adds rice_irrigated_area_1000ha column.
Note: irrigation_share still needs rice_area_sown (separate file).
"""

import pandas as pd
import numpy as np

FINAL_PANEL = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\final_panel.csv"
IRRIG_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\rice_irrigated_area_clean.csv"
OUT_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\final_panel.csv"

# Load
final = pd.read_csv(FINAL_PANEL)
irrig = pd.read_csv(IRRIG_PATH)
print(f"Final panel: {final.shape}, Irrigation: {irrig.shape}")

# Normalize district names for merge
for df in [final, irrig]:
    df["district_key"] = df["Dist Name"].str.strip().str.title()
    df["year_key"] = df["Year"].astype(int)

# Merge
merged = final.merge(
    irrig[["district_key", "year_key", "irrig_clean"]],
    on=["district_key", "year_key"],
    how="left",
    indicator=True
)
print(f"Merge result: {merged['_merge'].value_counts().to_dict()}")

# Rename and clean up
merged.rename(columns={"irrig_clean": "rice_irrigated_area_1000ha"}, inplace=True)
merged.drop(columns=["_merge", "district_key", "year_key"], inplace=True)

# irrigation_share placeholder (needs rice area sown)
merged["irrigation_share"] = np.nan

# Save
merged.to_csv(OUT_PATH, index=False)
print(f"\n✅ Updated final panel saved: {OUT_PATH}")
print(f"Shape: {merged.shape}")
print(f"Missing rice_irrigated_area_1000ha: {merged['rice_irrigated_area_1000ha'].isna().sum()}")
print(f"Non-null by state:")
print(merged.groupby("State Name")["rice_irrigated_area_1000ha"].apply(lambda x: x.notna().sum()))