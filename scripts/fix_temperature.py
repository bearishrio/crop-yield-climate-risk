"""
Fix temperature data for districts missing 1° grid cells using nearest-neighbor.
Updates final_panel.csv in place.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

CLIMATE_DIR = Path(r"C:\Users\rioth\OneDrive\Desktop\Test\yield\raw\climate")
TEMP_GRID = {"nlat": 31, "nlon": 31, "lat_start": 7.5, "lat_end": 37.5, "lon_start": 67.5, "lon_end": 97.5}
PANEL_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\final_panel.csv"

# Load final panel
panel = pd.read_csv(PANEL_PATH)
print(f"Panel shape: {panel.shape}")

# Load temp data
def read_grd_yearwise(var_dir, grid, var_name, start_year, end_year):
    var_dir = Path(var_dir)
    all_data, all_times = [], []
    for year in range(start_year, end_year + 1):
        f = var_dir / f"{year}.grd"
        if not f.exists(): continue
        data = np.fromfile(f, dtype='<f4')
        n_cells = grid["nlat"] * grid["nlon"]
        n_days = len(data) // n_cells
        if len(data) % n_cells != 0:
            data = data[:n_days * n_cells]
        data = data.reshape(n_days, grid["nlat"], grid["nlon"])
        all_data.append(data)
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')[:n_days]
        all_times.append(dates)
    combined = np.concatenate(all_data, axis=0)
    times = np.concatenate(all_times)
    lats = np.linspace(grid["lat_start"], grid["lat_end"], grid["nlat"])
    lons = np.linspace(grid["lon_start"], grid["lon_end"], grid["nlon"])
    return xr.DataArray(combined, dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": lats, "lon": lons}, name=var_name)

print("Loading temperature data...")
tmax_da = read_grd_yearwise(CLIMATE_DIR / "tmax", TEMP_GRID, "tmax", 1990, 2019)
tmin_da = read_grd_yearwise(CLIMATE_DIR / "tmin", TEMP_GRID, "tmin", 1990, 2019)

# Temp grid coordinates
tmax_lats = np.linspace(TEMP_GRID["lat_start"], TEMP_GRID["lat_end"], TEMP_GRID["nlat"])
tmax_lons = np.linspace(TEMP_GRID["lon_start"], TEMP_GRID["lon_end"], TEMP_GRID["nlon"])
time_vals = tmax_da.time.values  # numpy datetime64[ns]

# District approximate centroids (lat, lon)
DISTRICT_CENTROIDS = {
    "Amarawati": (20.9, 77.8), "Amritsar": (31.6, 74.9), "Angul": (20.8, 85.1),
    "Balasore": (21.5, 86.9), "Bhadrak": (21.1, 86.5), "Bhatinda": (30.2, 74.9),
    "Deogarh": (21.5, 84.7), "Dhenkanal": (20.7, 85.6), "Dhule": (20.9, 74.8),
    "Faridkot": (30.7, 74.8), "Fatehgarh Sahib": (30.6, 76.4), "Ferozpur": (30.9, 74.6),
    "Gajapati": (18.8, 84.2), "Gondia": (21.5, 80.2), "Gurdaspur": (32.0, 75.4),
    "Hoshiarpur": (31.5, 75.9), "Jagatsinghapur": (20.3, 86.2), "Jajapur": (20.8, 86.3),
    "Jalandhar": (31.3, 75.6), "Jalgaon": (21.0, 75.6), "Kapurthala": (31.4, 75.4),
    "Kendrapara": (20.5, 86.4), "Keonjhar": (21.6, 85.6), "Khurda": (20.1, 85.5),
    "Ludhiana": (30.9, 75.9), "Mansa": (29.9, 75.4), "Moga": (30.8, 75.2),
    "Muktsar": (30.5, 74.5), "Nawarangpur": (19.2, 82.5),
    "Phulbani(Kandhamal)": (20.5, 84.2), "Puri": (19.8, 85.8), "Rayagada": (19.2, 83.4),
    "Sambalpur": (21.5, 84.0), "Sangrur": (30.2, 75.8), "Satara": (17.7, 74.0),
    "Sonepur": (20.8, 83.9), "Sundargarh": (22.1, 84.0),
}

# Find missing districts in panel
missing_districts = panel[panel["tmax_kharif_mean_c"].isna()]["Dist Name"].unique()
print(f"Fixing {len(missing_districts)} districts...")

for d in missing_districts:
    if d not in DISTRICT_CENTROIDS:
        print(f"  No centroid for {d}, skipping")
        continue
    lat, lon = DISTRICT_CENTROIDS[d]
    lat_idx = np.abs(tmax_lats - lat).argmin()
    lon_idx = np.abs(tmax_lons - lon).argmin()
    print(f"  {d}: centroid ({lat:.2f}, {lon:.2f}) -> grid ({lat_idx},{lon_idx}) = ({tmax_lats[lat_idx]:.2f}, {tmax_lons[lon_idx]:.2f})")
    
    # Extract time series for this grid cell
    tmax_series = tmax_da.isel(lat=lat_idx, lon=lon_idx).values
    tmin_series = tmin_da.isel(lat=lat_idx, lon=lon_idx).values
    
    # Compute kharif features for each year using numpy datetime64
    for year in range(1990, 2020):
        # Month from datetime64: (year-month) encoded as months since epoch
        month = time_vals.astype('datetime64[M]').astype(int) % 12 + 1
        yr = time_vals.astype('datetime64[Y]').astype(int) + 1970
        kharif_mask = (yr == year) & (month >= 6) & (month <= 10)
        
        tmax_kharif = tmax_series[kharif_mask]
        tmin_kharif = tmin_series[kharif_mask]
        if len(tmax_kharif) == 0:
            continue
        tmax_mean = np.nanmean(tmax_kharif)
        tmin_mean = np.nanmean(tmin_kharif)
        temp_mean = (tmax_mean + tmin_mean) / 2
        extreme_heat = np.nansum(tmax_kharif > 35)
        
        # Update panel
        mask = (panel["Dist Name"] == d) & (panel["Year"] == year)
        if mask.any():
            panel.loc[mask, "tmax_kharif_mean_c"] = tmax_mean
            panel.loc[mask, "tmin_kharif_mean_c"] = tmin_mean
            panel.loc[mask, "temp_kharif_mean_c"] = temp_mean
            panel.loc[mask, "extreme_heat_days"] = extreme_heat

# Recompute temp_anomaly and extreme_heat_anomaly for all districts
print("\nRecomputing anomalies...")
baseline = panel[panel["Year"].between(1990, 2010)]
for col, anom_col in [("temp_kharif_mean_c", "temp_anomaly"), ("extreme_heat_days", "extreme_heat_anomaly")]:
    means = baseline.groupby("Dist Name")[col].mean()
    stds = baseline.groupby("Dist Name")[col].std().replace(0, np.nan)
    panel[anom_col] = panel.apply(
        lambda r: (r[col] - means.get(r["Dist Name"], np.nan)) / stds.get(r["Dist Name"], np.nan), axis=1)

# Save
panel.to_csv(PANEL_PATH, index=False)
print(f"\n✅ Updated panel saved: {PANEL_PATH}")
print(f"Missing tmax: {panel['tmax_kharif_mean_c'].isna().sum()}")
print(f"Missing temp_anomaly: {panel['temp_anomaly'].isna().sum()}")
print(f"Missing extreme_heat_anomaly: {panel['extreme_heat_anomaly'].isna().sum()}")