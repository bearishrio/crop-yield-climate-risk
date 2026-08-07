"""
Aggregate climate grids to district-level kharif features + merge with yield panel.
Uses local Census 2011 district shapefile + ICRISAT district name mapping.
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
CLIMATE_DIR = Path(r"C:\Users\rioth\OneDrive\Desktop\Test\yield\raw\climate")
YIELD_PATH = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\rice_yield_clean.csv"
OUTPUT_DIR = Path(r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMD grid specs
RAIN_GRID = {"nlat": 129, "nlon": 135, "lat_start": 6.5, "lat_end": 38.5, "lon_start": 66.5, "lon_end": 100.5}
TEMP_GRID = {"nlat": 31, "nlon": 31, "lat_start": 7.5, "lat_end": 37.5, "lon_start": 67.5, "lon_end": 97.5}

# Kharif season (Jun-Oct)
KHARIF_MONTHS = [6, 7, 8, 9, 10]
BASELINE_START, BASELINE_END = 1990, 2010

# ============================================================
# DISTRICT NAME MAPPING: Census 2011 -> ICRISAT
# ============================================================
CENSUS_TO_ICRISAT = {
    "Ahmadnagar": "Ahmednagar", "Amravati": "Amarawati", "Garhchiroli": "Gadchiroli",
    "Gondiya": "Gondia", "Nashik": "Nasik", "Yavatmal": "Yeotmal",
    "Bathinda": "Bhatinda", "Firozpur": "Ferozpur", "Muktsar": "Shri Mukatsar Sahib",
    "Rupnagar": "Roopnagar", "Sahibzada Ajit Singh Nagar": "S.B.S Nagar",
    "Shahid Bhagat Singh Nagar": "S.B.S Nagar",
    "Anugul": "Angul", "Balangir": "Bolangir", "Baleshwar": "Balasore", "Bauda": "Boudh",
    "Debagarh": "Deogarh", "Kandhamal": "Phulbani(Kandhamal)", "Kendujhar": "Keonjhar",
    "Khordha": "Khurda", "Nabarangapur": "Nawarangpur", "Subarnapur": "Sonepur",
    "Mayurbhanj": "Mayurbhanja",
}

# Target ICRISAT districts (74 districts from cleaned yield panel)
ICRISAT_DISTRICTS = {
    "Maharashtra": ["Ahmednagar", "Amarawati", "Aurangabad", "Bhandara", "Bid", "Chandrapur",
                    "Dhule", "Gadchiroli", "Gondia", "Jalgaon", "Kolhapur", "Latur", "Nagpur",
                    "Nanded", "Nandurbar", "Nasik", "Osmanabad", "Parbhani", "Pune", "Raigarh",
                    "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Yeotmal"],
    "Punjab": ["Amritsar", "Bhatinda", "Faridkot", "Fatehgarh Sahib", "Ferozpur", "Gurdaspur",
               "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Patiala",
               "Roopnagar", "S.B.S Nagar", "Sangrur", "Shri Mukatsar Sahib"],
    "Orissa": ["Angul", "Balasore", "Bargarh", "Bhadrak", "Bolangir", "Boudh", "Cuttack",
               "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghapur", "Jajapur",
               "Jharsuguda", "Kalahandi", "Kendrapara", "Keonjhar", "Khurda", "Koraput",
               "Malkangiri", "Mayurbhanja", "Nawarangpur", "Nayagarh", "Nuapada",
               "Phulbani(Kandhamal)", "Puri", "Rayagada", "Sambalpur", "Sonepur", "Sundargarh"]
}
ALL_ICRISAT = [d for lst in ICRISAT_DISTRICTS.values() for d in lst]

# ============================================================
# 1. LOAD CLIMATE DATA
# ============================================================
def read_grd_yearwise(var_dir, grid, var_name, start_year, end_year):
    var_dir = Path(var_dir)
    all_data, all_times = [], []
    for year in range(start_year, end_year + 1):
        f = var_dir / f"{year}.grd"
        if not f.exists():
            print(f"  Missing: {f}")
            continue
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

print("Loading climate data...")
rain_da = read_grd_yearwise(CLIMATE_DIR / "rain", RAIN_GRID, "rainfall", 1990, 2019)
tmax_da = read_grd_yearwise(CLIMATE_DIR / "tmax", TEMP_GRID, "tmax", 1990, 2019)
tmin_da = read_grd_yearwise(CLIMATE_DIR / "tmin", TEMP_GRID, "tmin", 1990, 2019)
print(f"  Rain: {rain_da.shape}, Tmax: {tmax_da.shape}, Tmin: {tmin_da.shape}")

# ============================================================
# 2. LOAD DISTRICT BOUNDARIES (Census 2011)
# ============================================================
print("\nLoading district boundaries (Census 2011)...")
boundaries_path = Path(r"C:\Users\rioth\OneDrive\Desktop\Test\data\raw\boundaries\India-Districts-2011Census.shp")
districts = gpd.read_file(boundaries_path)

# Filter to our 3 states
state_map = {"Maharashtra": "Maharashtra", "Punjab": "Punjab", "Orissa": "Odisha"}
districts = districts[districts["ST_NM"].isin(state_map.values())].copy()
print(f"  Districts in target states: {len(districts)}")

# Ensure CRS is WGS84
if districts.crs is None:
    districts.set_crs(epsg=4326, inplace=True)
else:
    districts = districts.to_crs(epsg=4326)

# Map Census district names -> ICRISAT names
districts["ICRISAT_NAME"] = districts["DISTRICT"].map(CENSUS_TO_ICRISAT).fillna(districts["DISTRICT"])

# Keep only districts that are in our ICRISAT target list
districts = districts[districts["ICRISAT_NAME"].isin(ALL_ICRISAT)].copy()
print(f"  Districts matching ICRISAT panel: {len(districts)}")

# ============================================================
# 3. SPATIAL AGGREGATION: GRID CELLS -> DISTRICTS (by ICRISAT name)
# ============================================================
print("\nBuilding grid cell -> district mapping...")

def grid_centroids(grid):
    lats = np.linspace(grid["lat_start"], grid["lat_end"], grid["nlat"])
    lons = np.linspace(grid["lon_start"], grid["lon_end"], grid["nlon"])
    points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            points.append({"lat_idx": i, "lon_idx": j, "lat": lat, "lon": lon,
                          "geometry": gpd.points_from_xy([lon], [lat])[0]})
    return gpd.GeoDataFrame(points, crs="EPSG:4326")

rain_pts = grid_centroids(RAIN_GRID)
temp_pts = grid_centroids(TEMP_GRID)

# Spatial join: which district contains each grid centroid?
rain_join = gpd.sjoin(rain_pts, districts[["ICRISAT_NAME", "geometry"]], how="left", predicate="within")
temp_join = gpd.sjoin(temp_pts, districts[["ICRISAT_NAME", "geometry"]], how="left", predicate="within")

# Drop grid cells not in any target district
rain_join = rain_join.dropna(subset=["ICRISAT_NAME"])
temp_join = temp_join.dropna(subset=["ICRISAT_NAME"])

print(f"  Rain grid cells matched: {len(rain_join)}")
print(f"  Temp grid cells matched: {len(temp_join)}")

# ============================================================
# 4. COMPUTE DAILY DISTRICT SERIES (only for ICRISAT districts)
# ============================================================
print("\nAggregating daily climate to district level...")

def aggregate_to_district(da, join_df, grid_nlat, grid_nlon, target_districts):
    """da: (time, lat, lon), join_df: grid_idx -> district mapping"""
    n_time = da.shape[0]
    results = {d: [] for d in target_districts}
    cell_idx = {d: [] for d in target_districts}
    for _, row in join_df.iterrows():
        d = row["ICRISAT_NAME"]
        if d in target_districts:
            idx = row["lat_idx"] * grid_nlon + row["lon_idx"]
            cell_idx[d].append(idx)
    data_flat = da.values.reshape(n_time, -1)
    for d in target_districts:
        if cell_idx[d]:
            results[d] = data_flat[:, cell_idx[d]].mean(axis=1)
        else:
            results[d] = np.full(n_time, np.nan)
    return pd.DataFrame(results, index=da.time.values)

rain_daily = aggregate_to_district(rain_da, rain_join, RAIN_GRID["nlat"], RAIN_GRID["nlon"], ALL_ICRISAT)
tmax_daily = aggregate_to_district(tmax_da, temp_join, TEMP_GRID["nlat"], TEMP_GRID["nlon"], ALL_ICRISAT)
tmin_daily = aggregate_to_district(tmin_da, temp_join, TEMP_GRID["nlat"], TEMP_GRID["nlon"], ALL_ICRISAT)

print(f"  Rain daily shape: {rain_daily.shape} (districts x time)")
print(f"  Tmax daily shape: {tmax_daily.shape}")

# ============================================================
# 5. KHARIF SEASON AGGREGATION (Jun-Oct per year)
# ============================================================
print("\nComputing kharif (Jun-Oct) seasonal features...")

def kharif_features(rain_df, tmax_df, tmin_df, target_districts):
    records = []
    for year in range(1990, 2020):
        start = pd.Timestamp(f"{year}-06-01")
        end = pd.Timestamp(f"{year}-10-31")
        rain_season = rain_df.loc[start:end]
        tmax_season = tmax_df.loc[start:end]
        tmin_season = tmin_df.loc[start:end]
        for d in target_districts:
            r = rain_season[d].values
            tx = tmax_season[d].values
            tn = tmin_season[d].values
            if len(r) == 0 or np.all(np.isnan(r)):
                continue
            rain_total = np.nansum(r)
            dry = (r < 2.5).astype(int)
            dry_spell = max_dry = 0
            for val in dry:
                if val == 1:
                    dry_spell += 1
                    max_dry = max(max_dry, dry_spell)
                else:
                    dry_spell = 0
            tmax_mean = np.nanmean(tx)
            tmin_mean = np.nanmean(tn)
            temp_mean = (tmax_mean + tmin_mean) / 2
            extreme_heat = np.nansum(tx > 35)
            records.append({
                "district_name": d, "year": year,
                "rainfall_kharif_mm": rain_total,
                "tmax_kharif_mean_c": tmax_mean,
                "tmin_kharif_mean_c": tmin_mean,
                "temp_kharif_mean_c": temp_mean,
                "extreme_heat_days": extreme_heat,
                "max_dry_spell_days": max_dry
            })
    return pd.DataFrame(records)

climate_features = kharif_features(rain_daily, tmax_daily, tmin_daily, ALL_ICRISAT)
print(f"  Climate features: {climate_features.shape}")

# ============================================================
# 6. ANOMALIES (standardized vs 1990-2010 baseline)
# ============================================================
print("\nComputing anomalies (baseline 1990-2010)...")
baseline = climate_features[climate_features["year"].between(BASELINE_START, BASELINE_END)]
anom_cols = ["rainfall_kharif_mm", "temp_kharif_mean_c", "extreme_heat_days", "max_dry_spell_days"]
for col in anom_cols:
    means = baseline.groupby("district_name")[col].mean()
    stds = baseline.groupby("district_name")[col].std().replace(0, np.nan)
    climate_features[f"{col}_anomaly"] = climate_features.apply(
        lambda r: (r[col] - means.get(r["district_name"], np.nan)) / stds.get(r["district_name"], np.nan), axis=1)

climate_features.rename(columns={
    "rainfall_kharif_mm_anomaly": "rainfall_anomaly",
    "temp_kharif_mean_c_anomaly": "temp_anomaly",
    "extreme_heat_days_anomaly": "extreme_heat_anomaly",
    "max_dry_spell_days_anomaly": "dry_spell_anomaly"
}, inplace=True)

# ============================================================
# 7. LOAD YIELD PANEL & MERGE
# ============================================================
print("\nLoading yield panel and merging...")
yield_df = pd.read_csv(YIELD_PATH)
print(f"  Yield rows: {len(yield_df)}, districts: {yield_df['Dist Name'].nunique()}")

# Normalize district names
for df in [yield_df, climate_features]:
    if "Dist Name" in df.columns:
        df["district_key"] = df["Dist Name"].str.strip().str.title()
    elif "district_name" in df.columns:
        df["district_key"] = df["district_name"].str.strip().str.title()

merged = yield_df.merge(
    climate_features,
    left_on=["district_key", "Year"],
    right_on=["district_key", "year"],
    how="left", indicator=True
)
print(f"  Merge result: {merged['_merge'].value_counts().to_dict()}")
merged.drop(columns=["_merge", "year", "district_key"], inplace=True)

# ============================================================
# 8. MERGE IRRIGATION DATA
# ============================================================
print("\nMerging irrigation data...")
irrig_path = r"C:\Users\rioth\OneDrive\Desktop\Test\data\processed\rice_irrigated_area_clean.csv"
if Path(irrig_path).exists():
    irrig = pd.read_csv(irrig_path)
    for df in [merged, irrig]:
        df["district_key"] = df["Dist Name"].str.strip().str.title()
        df["year_key"] = df["Year"].astype(int)
    merged = merged.merge(
        irrig[["district_key", "year_key", "irrig_clean"]],
        on=["district_key", "year_key"], how="left"
    )
    merged.rename(columns={"irrig_clean": "rice_irrigated_area_1000ha"}, inplace=True)
    merged.drop(columns=["district_key", "year_key"], inplace=True)
    print(f"  Irrigation merged. Non-null: {merged['rice_irrigated_area_1000ha'].notna().sum()}")
else:
    print("  Irrigation file not found, skipping.")
    merged["rice_irrigated_area_1000ha"] = np.nan

# irrigation_share placeholder (needs rice area sown)
merged["irrigation_share"] = np.nan

# ============================================================
# 9. LAGGED YIELD
# ============================================================
merged = merged.sort_values(["Dist Name", "Year"])
merged["lagged_yield"] = merged.groupby("Dist Name")["yield_clean"].shift(1)

# ============================================================
# 10. FINAL CLEANUP & SAVE
# ============================================================
final_cols = [
    "Dist Code", "Year", "State Code", "State Name", "Dist Name",
    "yield_clean",  # target
    "rainfall_kharif_mm", "rainfall_anomaly",
    "tmax_kharif_mean_c", "tmin_kharif_mean_c", "temp_kharif_mean_c", "temp_anomaly",
    "extreme_heat_days", "extreme_heat_anomaly",
    "max_dry_spell_days", "dry_spell_anomaly",
    "rice_irrigated_area_1000ha", "irrigation_share", "lagged_yield"
]
final = merged[final_cols].copy()
final.rename(columns={"yield_clean": "yield_kg_per_ha"}, inplace=True)

out_path = OUTPUT_DIR / "final_panel.csv"
final.to_csv(out_path, index=False)
print(f"\n✅ Saved final panel: {out_path}")
print(f"   Shape: {final.shape}")
print(f"   Missing per column:\n{final.isnull().sum()}")
print(f"   Years: {final['Year'].min()}-{final['Year'].max()}")
print(f"   Districts: {final['Dist Name'].nunique()}")

print("\nSample rows:")
print(final[["Dist Name", "Year", "yield_kg_per_ha", "rainfall_anomaly", "temp_anomaly"]].head(10))