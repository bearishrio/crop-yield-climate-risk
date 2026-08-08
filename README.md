# Crop Yield Climate Risk Project
# Predicting Crop Yield Risk from Climate Variables in India

Analysis of rice yield sensitivity to climate variability across 74 districts
in Maharashtra, Odisha, and Punjab (1990–2019), using panel regression and
machine learning to construct a district-level climate risk ranking with
implications for crop insurance and irrigation policy targeting.

## Key Finding

Irrigation coverage buffers rice yield against rainfall shocks — but
nonlinearly. Districts with near-complete irrigation (Punjab, ~99% coverage)
show almost no predicted yield loss from a 1-standard-deviation rainfall
deficit, while partially irrigated districts (Odisha, ~40% average coverage)
show sensitivity nearly as high as fully rainfed districts (Maharashtra).
This pattern holds consistently across two independent modeling approaches.

## Methodology

- **Data:** ICRISAT district-level rice yield, area, and irrigation data;
  IMD gridded rainfall (0.25°) and temperature (1°) data, 1990–2019
- **Models:** OLS regression with district and year fixed effects; Random
  Forest regressor, validated via a temporal holdout (train 1991–2009, test
  2010–2019) and district-grouped 5-fold cross-validation
- **Output:** a district-level climate sensitivity ranking, simulating
  predicted yield impact under a 1-SD rainfall deficit

## Repository Structure

- `scripts/` — full data pipeline: cleaning, climate aggregation, modeling
- `data/raw/` — original ICRISAT and boundary files
- `data/processed/` — cleaned panel datasets and the final sensitivity ranking
- `outputs/` — regression results and figures
- `report/` — full write-up (LaTeX)

## Reproducing This

```bash
conda create -n cropyield python=3.11
conda activate cropyield
pip install -r requirements.txt
```

Run the scripts in `scripts/` in this order: `clean_icrisat_rice.py` →
`clean_icrisat_irrigation.py` → `download_climate.py` → `aggregate_climate.py`
→ `fix_temperature.py` → `fix_remaining_temp.py` → `add_irrigation_share.py`
→ `baseline_regression.py` → `modeling.py`

## Data Sources

- [ICRISAT District Level Database](http://data.icrisat.org/dld/)
- [IMD Gridded Climate Data](https://imdpune.gov.in/cmpg/Griddata/)
