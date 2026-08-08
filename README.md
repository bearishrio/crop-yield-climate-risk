# Climate Risk and Rice Yield Prediction in India

This project estimates how climate variability is associated with rice yields
across 74 districts in Maharashtra, Odisha, and Punjab, India, over 1990–2019.
It combines district-level agricultural data with gridded rainfall and
temperature data, compares a fixed-effects OLS model with a random forest, and
produces a district-level climate-sensitivity ranking.

The intended use is research and policy analysis: identifying where rainfall
shortfalls may create greater predicted yield losses and where irrigation may
buffer that risk. Model outputs are predictive scenario estimates, not causal
treatment effects.

## Main findings from the current model run

- Temporal holdout: training 1991–2009; testing 2010–2019.
- Random forest test performance on the original kg/ha scale:
  R-squared = 0.754 and RMSE = 614.005 kg/ha.
- Forecasting-valid OLS test performance: R-squared = 0.691 and
  RMSE = 688.034 kg/ha.
- Naive persistence benchmark (current yield = lagged yield):
  R-squared = 0.716 and RMSE = 660.318 kg/ha.
- Random-forest feature importance: log lagged yield 0.722, irrigation share
  0.115, and rainfall anomaly 0.067.
- Five-fold district-grouped cross-validation: mean log-yield R-squared =
  0.719, standard deviation = 0.206. The lowest-performing fold had
  R-squared = 0.316 and contained 15 Maharashtra and Odisha districts.
- State-level rainfall-only scenario averages: Maharashtra 19.491% predicted
  loss, Odisha 19.263%, and Punjab -0.451%. Punjab's near-complete irrigation
  is consistent with lower modeled rainfall sensitivity, although this is not
  causal evidence.

## Data and variables

The final panel is located at:

~~~text
data/processed/final_panel.csv
~~~

It contains 2,064 raw district-year rows before modeling cleanup. The modeling
sample has 1,990 observations after removing 74 first-year rows with no
lagged yield.

| Variable | Description |
| --- | --- |
| Dist Name | District identifier |
| State Name | State identifier |
| Year | 1990–2019 |
| yield_kg_per_ha | Rice yield outcome |
| rainfall_anomaly | District-standardized rainfall anomaly |
| temp_anomaly | District-standardized temperature anomaly |
| extreme_heat_anomaly | District-standardized extreme-heat anomaly |
| dry_spell_anomaly | District-standardized dry-spell anomaly |
| irrigation_share | Irrigated rice area divided by rice area |
| lagged_yield | Prior-year district yield |

Climate anomalies are standardized relative to each district's own historical
baseline. Yield and lagged yield are log-transformed for modeling. Missing
irrigation_share observations in Maharashtra are filled with zero according to
the project's rainfed-district assumption; this should be validated if better
irrigation data become available.

Sources:

- [ICRISAT District Level Database](http://data.icrisat.org/dld/) for district
  rice yield, area, and irrigation inputs.
- [IMD gridded climate data](https://imdpune.gov.in/cmpg/Griddata/) for
  rainfall and temperature.

Raw climate grids are ignored by Git because of their size and are stored
locally under yield/raw/climate/ when available.

## Modeling workflow

The fitted models predict log(yield_kg_per_ha) using:

~~~text
rainfall_anomaly
temp_anomaly
extreme_heat_anomaly
dry_spell_anomaly
irrigation_share
log(lagged_yield)
~~~

The 1990 rows are dropped because each district's first observation has no
prior-year yield. The year-based split better represents forecasting later
years than a random split, which could place related adjacent years in both
training and test data.

The OLS baseline includes the climate features, irrigation, log lagged yield,
district fixed effects, and year fixed effects. For the forecasting comparison,
OLS is refit on the training period only. Test-year fixed effects are
unavailable without using test outcomes, so the forecast uses the average
estimated training-period year effect.

The random forest uses 500 trees, minimum leaf size 2, random_state 42, and
all available processor cores. It is evaluated on the temporal holdout and
five-fold GroupKFold validation grouped by district.

## Climate-sensitivity ranking

modeling.py creates:

~~~text
data/processed/district_climate_sensitivity_ranking.csv
~~~

For each district, it predicts yield under:

1. Normal rainfall: rainfall_anomaly = 0.
2. Rainfall-only shock: rainfall_anomaly = -1.
3. Compound shock: rainfall_anomaly = -1 and temp_anomaly = +1.

All other features are held at the district's historical training-period
average. Predictions are made in log-yield space and exponentiated back to
kg/ha. The rainfall-only loss is:

~~~text
100 * (predicted_normal_yield - predicted_rainfall_shock_yield)
    / predicted_normal_yield
~~~

The compound columns are predicted_yield_loss_compound_pct and
compound_vs_rainfall_only_gap_pct. The ranking also includes
low_performing_cv_fold, which flags districts from the weakest
district-grouped validation fold.

These scenarios are model-based counterfactual predictions. They do not
identify causal effects, and fixed-feature scenarios may combine climate
conditions that were uncommon in a district's historical data.

## Figures

Run:

~~~powershell
python scripts/visualize_ranking.py
~~~

This creates:

- outputs/figures/climate_sensitivity_ranking.png
- outputs/figures/climate_sensitivity_ranking.pdf

The figure contains a rainfall-only top-15 district ranking, with state colors
and hatching for low-performing validation-fold districts. If compound
predictions are available, it also contains a state-average comparison of
rainfall-only and compound losses.

visualize_results.py creates the broader four-panel project summary:

- outputs/figures/project_dashboard.png
- outputs/figures/project_dashboard.pdf

## Interpretation and limitations

Feature importance indicates predictive usefulness inside the random forest.
It does not provide a causal effect, a marginal yield response, a sign, or a
statistical significance test. Correlated climate variables can share or shift
importance.

The rainfall-only and compound rankings should be interpreted as risk-screening
tools. The compound scenario holds extreme heat and dry-spell anomalies fixed
to isolate the added temperature component. A richer scenario could jointly
shock rainfall, temperature, extreme heat, and dry spells using their observed
multivariate distribution, while allowing irrigation and management responses
to change.

The source data may use the state label Orissa; figures and presentation
outputs rename it to Odisha.
