"""
Fix remaining 11 districts missing temperature data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

CLIMATE_DIR = Path(r"C:\Users\rioth\OneDrive\Desktop\Test\yield\raw\climate")
TEMP_GRID = {"nlat": 31, "nlon": 31, "lat_start": 7.5, "lat_end": 37.5, "lon_start": 67.5, "lon_end": 97.5}
PANEL_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\final_panel.csv"

panel = pd.read_csv(PANEL_PATH)

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

tmax_da = read_grd_yearwise(CLIMATE_DIR / "tmax", TEMP_GRID, "tmax", 1990, 2019)
tmin_da = read_grd_yearwise(CLIMATE_DIR / "tmin", TEMP_GRID, "tmin", 1990, 2019)

tmax_lats = np.linspace(TEMP_GRID["lat_start"], TEMP_GRID["lat_end"], TEMP_GRID["nlat"])
tmax_lons = np.linspace(TEMP_GRID["lon_start"], TEMP_GRID["lon_end"], TEMP_GRID["nlon"])
time_vals = tmax_da.time.values

# Remaining 11 districts
DISTRICT_CENTROIDS = {
    "Jharsuguda": (21.8, 84.0), "Kalahandi": (20.1, 83.5), "Kolhapur": (16.7, 74.2),
    "Koraput": (18.8, 82.7), "Latur": (18.4, 76.6), "Malkangiri": (18.3, 81.9),
    "Nayagarh": (20.1, 85.1), "Raigarh": (21.9, 83.4), "Roopnagar": (30.9, 76.5),
    "S.B.S Nagar": (31.1, 76.1), "Sangli": (16.9, 74.6),
}

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
    
    tmax_series = tmax_da.isel(lat=lat_idx, lon=lon_idx).values
    tmin_series = tmin_da.isel(lat=lat_idx, lon=lon_idx).values
    
    for year in range(1990, 2020):
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
        
        mask = (panel["Dist Name"] == d) & (panel["Year"] == year)
        if mask.any():
            panel.loc[mask, "tmax_kharif_mean_c"] = tmax_mean
            panel.loc[mask, "tmin_kharif_mean_c"] = tmin_mean
            panel.loc[mask, "temp_kharif_mean_c"] = temp_mean
            panel.loc[mask, "extreme_heat_days"] = extreme_heat

# Recompute anomalies
baseline = panel[panel["Year"].between(1990, 2010)]
for col, anom_col in [("temp_kharif_mean_c", "temp_anomaly"), ("extreme_heat_days", "extreme_heat_anomaly")]:
    means = baseline.groupby("Dist Name")[col].mean()
    stds = baseline.groupby("Dist Name")[col].std().replace(0, np.nan)
    panel[anom_col] = panel.apply(
        lambda r: (r[col] - means.get(r["Dist Name"], np.nan)) / stds.get(r["Dist Name"], np.nan), axis=1)

panel.to_csv(PANEL_PATH, index=False)
print(f"Done. Missing tmax: {panel['tmax_kharif_mean_c'].isna().sum()}")