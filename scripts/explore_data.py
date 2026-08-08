"""Explore climate + yield data - custom reader for downloaded .grd files."""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# --- Config ---
CLIMATE_DIR = Path(r"C:\Users\rioth\OneDrive\Desktop\Test\yield\raw\climate")
YIELD_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\rice_yield_clean.csv"

# IMD grid specs (from imdlib source)
RAIN_GRID = {"nlat": 129, "nlon": 135, "lat_start": 6.5, "lat_end": 38.5, "lon_start": 66.5, "lon_end": 100.5}
TEMP_GRID = {"nlat": 31, "nlon": 31, "lat_start": 7.5, "lat_end": 37.5, "lon_start": 67.5, "lon_end": 97.5}

def read_grd_yearwise(var_dir, grid, var_name, start_year, end_year):
    """Read yearwise .grd files (named 1990.grd, 1991.grd, ...) into xarray DataArray."""
    var_dir = Path(var_dir)
    all_data = []
    all_times = []
    
    for year in range(start_year, end_year + 1):
        f = var_dir / f"{year}.grd"
        if not f.exists():
            print(f"  Missing: {f}")
            continue
        
        # Read binary: float32, little-endian, shape (nlat, nlon) per day
        data = np.fromfile(f, dtype='<f4')
        n_cells = grid["nlat"] * grid["nlon"]
        n_days = len(data) // n_cells
        
        if len(data) % n_cells != 0:
            print(f"  Warning: {f} size not divisible by grid cells ({len(data)} % {n_cells} = {len(data) % n_cells})")
            data = data[:n_days * n_cells]
        
        data = data.reshape(n_days, grid["nlat"], grid["nlon"])
        all_data.append(data)
        
        # Daily timestamps for this year
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
        if len(dates) != n_days:
            # Handle leap year mismatch
            dates = dates[:n_days]
        all_times.append(dates)
    
    if not all_data:
        raise ValueError(f"No data found in {var_dir}")
    
    combined = np.concatenate(all_data, axis=0)
    times = np.concatenate(all_times)
    
    lats = np.linspace(grid["lat_start"], grid["lat_end"], grid["nlat"])
    lons = np.linspace(grid["lon_start"], grid["lon_end"], grid["nlon"])
    
    da = xr.DataArray(
        combined,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
        name=var_name
    )
    return da

# --- Load climate data ---
print("=== LOADING CLIMATE DATA ===")

print("\n--- RAIN (0.25°) ---")
rain_da = read_grd_yearwise(CLIMATE_DIR / "rain", RAIN_GRID, "rainfall", 1990, 2019)
print(f"Shape (time, lat, lon): {rain_da.shape}")
print(f"Time range: {rain_da.time.values[0]} to {rain_da.time.values[-1]}")
print(f"Lat range: {rain_da.lat.values[0]:.2f} to {rain_da.lat.values[-1]:.2f}")
print(f"Lon range: {rain_da.lon.values[0]:.2f} to {rain_da.lon.values[-1]:.2f}")

print("\n--- TMAX (1°) ---")
tmax_da = read_grd_yearwise(CLIMATE_DIR / "tmax", TEMP_GRID, "tmax", 1990, 2019)
print(f"Shape (time, lat, lon): {tmax_da.shape}")
print(f"Time range: {tmax_da.time.values[0]} to {tmax_da.time.values[-1]}")

print("\n--- TMIN (1°) ---")
tmin_da = read_grd_yearwise(CLIMATE_DIR / "tmin", TEMP_GRID, "tmin", 1990, 2019)
print(f"Shape (time, lat, lon): {tmin_da.shape}")
print(f"Time range: {tmin_da.time.values[0]} to {tmin_da.time.values[-1]}")

# --- Load yield data ---
print(f"\n=== LOADING YIELD DATA ===")
print(f"Path: {YIELD_PATH}")

yield_df = pd.read_csv(YIELD_PATH)
print(f"\nColumns: {yield_df.columns.tolist()}")
print(f"Shape: {yield_df.shape}")
print(f"\nFirst 5 rows:")
print(yield_df.head())
print(f"\nYears: {sorted(yield_df['Year'].unique())}")
print(f"States: {yield_df['State Name'].unique()}")
print(f"Crops: {'RICE (already filtered)' if 'RICE YIELD (Kg per ha)' in yield_df.columns else 'N/A'}")

# District/year coverage per state
if 'State Name' in yield_df.columns and 'Dist Name' in yield_df.columns:
    print(f"\n=== COVERAGE PER STATE ===")
    cov = yield_df.groupby('State Name').agg(
        districts=('Dist Name', 'nunique'),
        years=('Year', 'nunique'),
        rows=('Year', 'count')
    ).sort_values('districts', ascending=False)
    print(cov.to_string())
else:
    print("\nCould not compute coverage - missing 'State Name' or 'Dist Name' column")