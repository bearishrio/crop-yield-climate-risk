# modeling.py
# Compare Random Forest, forecasting-valid OLS fixed effects, and naive baseline

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ------------------------------------------------------------
# 1. Load and prepare the panel data
# ------------------------------------------------------------
panel = pd.read_csv("data/processed/final_panel.csv")

print("Original dataset shape:", panel.shape)
print("Year range:", panel["Year"].min(), "to", panel["Year"].max())

panel_model = panel.copy()

# Missing irrigation_share represents rainfed districts with zero irrigation.
panel_model["irrigation_share"] = panel_model["irrigation_share"].fillna(0)

# Remove each district's first observation because it has no lagged yield.
rows_before = len(panel_model)
panel_model = panel_model.dropna(subset=["lagged_yield"]).copy()
rows_dropped = rows_before - len(panel_model)

print(f"Rows dropped because lagged_yield is missing: {rows_dropped}")
print(f"Rows remaining for modeling: {len(panel_model)}")

# Log transforms require strictly positive values.
assert (panel_model["yield_kg_per_ha"] > 0).all(), (
    "yield_kg_per_ha must be positive."
)
assert (panel_model["lagged_yield"] > 0).all(), (
    "lagged_yield must be positive."
)

panel_model["log_yield"] = np.log(panel_model["yield_kg_per_ha"])
panel_model["log_lagged_yield"] = np.log(panel_model["lagged_yield"])

feature_cols = [
    "rainfall_anomaly",
    "temp_anomaly",
    "extreme_heat_anomaly",
    "dry_spell_anomaly",
    "irrigation_share",
    "log_lagged_yield",
]

# Confirm no missing values in variables used by the models.
model_cols = [
    "Year",
    "Dist Name",
    "yield_kg_per_ha",
    "log_yield",
    *feature_cols,
]

missing_values = panel_model[model_cols].isna().sum()
print("\nMissing values in modeling variables:")
print(missing_values)

assert missing_values.sum() == 0, "Missing values remain in modeling data."


# ------------------------------------------------------------
# 2. Year-based train/test split
# ------------------------------------------------------------
train_mask = panel_model["Year"].between(1990, 2009)
test_mask = panel_model["Year"].between(2010, 2019)

train_data = panel_model.loc[train_mask].copy()
test_data = panel_model.loc[test_mask].copy()

print("\nSplit summary:")
print(
    f"Training years: {train_data['Year'].min()}–{train_data['Year'].max()}"
)
print(
    f"Testing years:  {test_data['Year'].min()}–{test_data['Year'].max()}"
)
print(f"Training rows:  {len(train_data)}")
print(f"Testing rows:   {len(test_data)}")


# ------------------------------------------------------------
# 3. Fit Random Forest
# ------------------------------------------------------------
X_train = train_data[feature_cols].copy()
y_train = train_data["log_yield"].copy()

X_test = test_data[feature_cols].copy()
y_test_log = test_data["log_yield"].copy()
y_test_kg = test_data["yield_kg_per_ha"].copy()

rf_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_train_r2_log = rf_model.score(X_train, y_train)
rf_test_r2_log = rf_model.score(X_test, y_test_log)

print("\nRandom Forest performance on log-yield scale:")
print(
    pd.DataFrame(
        {
            "Period": ["Train (1991–2009)", "Test (2010–2019)"],
            "R²": [rf_train_r2_log, rf_test_r2_log],
        }
    ).round(3)
)

# Convert Random Forest test predictions from log yield to kg/ha.
rf_pred_log = rf_model.predict(X_test)
rf_pred_kg = np.exp(rf_pred_log)


# ------------------------------------------------------------
# 4. Fit OLS with district and year fixed effects on training data
# ------------------------------------------------------------
# Check every test district was also seen in the training data, so its
# district fixed effect can be used for prediction.
unseen_districts = set(test_data["Dist Name"]) - set(train_data["Dist Name"])

assert not unseen_districts, (
    f"Test districts missing from training data: {unseen_districts}"
)

ols_formula = """
log_yield ~ rainfall_anomaly
          + temp_anomaly
          + extreme_heat_anomaly
          + dry_spell_anomaly
          + irrigation_share
          + log_lagged_yield
          + C(Q("Dist Name"))
          + C(Year)
"""

ols_model = smf.ols(formula=ols_formula, data=train_data).fit()

print("\nOLS training R² on log-yield scale:")
print(round(ols_model.rsquared, 3))


# ------------------------------------------------------------
# 5. Forecast 2010-2019 with OLS
# ------------------------------------------------------------
# District effects are available for the test districts.
# But 2010-2019 year effects were not estimated during training.
#
# Use the average estimated training-period year effect as a neutral,
# forecasting-valid assumption. This avoids using test-period outcomes.

reference_year = train_data["Year"].min()

year_fe_terms = ols_model.params[
    ols_model.params.index.str.contains(r"C\(Year\)")
]

# Include the reference year's implicit fixed effect of zero.
average_train_year_effect = (
    year_fe_terms.sum() / train_data["Year"].nunique()
)

# Set test years to the known reference year for the design matrix,
# then add the average historical year effect.
ols_test_for_prediction = test_data.copy()
ols_test_for_prediction["Year"] = reference_year

ols_pred_log = (
    ols_model.predict(ols_test_for_prediction)
    + average_train_year_effect
)

ols_pred_kg = np.exp(ols_pred_log)


# ------------------------------------------------------------
# 6. Naive baseline: this year's yield = last year's yield
# ------------------------------------------------------------
naive_pred_kg = test_data["lagged_yield"].copy()


# ------------------------------------------------------------
# 7. Compare all models on the 2010-2019 test set in kg/ha
# ------------------------------------------------------------
def calculate_metrics(actual, predicted):
    return {
        "Test R² (kg/ha scale)": r2_score(actual, predicted),
        "Test RMSE (kg/ha)": np.sqrt(mean_squared_error(actual, predicted)),
    }


performance_comparison = pd.DataFrame([
    {
        "Model": "OLS: district + year FE",
        **calculate_metrics(y_test_kg, ols_pred_kg),
    },
    {
        "Model": "Random forest",
        **calculate_metrics(y_test_kg, rf_pred_kg),
    },
    {
        "Model": "Naive: yield = lagged_yield",
        **calculate_metrics(y_test_kg, naive_pred_kg),
    },
])

print("\n2010–2019 out-of-sample performance:")
print(performance_comparison.round(3))

# ------------------------------------------------------------
# 8. Random Forest feature importance plot
# ------------------------------------------------------------
import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    "Feature": rf_model.feature_names_in_,
    "Importance": rf_model.feature_importances_,
}).sort_values("Importance", ascending=True)

print("\nRandom Forest feature importances:")
print(importance_df.sort_values("Importance", ascending=False).round(3))

plt.figure(figsize=(9, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue"
)

plt.xlabel("Random Forest feature importance")
plt.ylabel("Feature")
plt.title("Feature importance for predicting rice yield risk")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Cross-validation: GroupKFold by district
# ------------------------------------------------------------
from sklearn.model_selection import GroupKFold, cross_val_score

# Each district is assigned entirely to one validation fold.
# No district can appear in both the training and validation portions of a fold.
groups_train = train_data["Dist Name"]

group_cv = GroupKFold(n_splits=5)

# Use a fresh model definition; cross_val_score refits it in every fold.
rf_cv_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

# R² is evaluated on log_yield, matching the earlier RF train/test R² output.
cv_r2_scores = cross_val_score(
    rf_cv_model,
    X_train,
    y_train,
    groups=groups_train,
    cv=group_cv,
    scoring="r2",
)

print("\n5-fold GroupKFold cross-validation results (grouped by district):")
print("Fold R² scores:", np.round(cv_r2_scores, 3))
print(f"Mean CV R²: {cv_r2_scores.mean():.3f}")
print(f"CV R² standard deviation: {cv_r2_scores.std():.3f}")

print(f"\nSingle future-period test R² (2010–2019): {rf_test_r2_log:.3f}")

# ------------------------------------------------------------
# Identify and describe the lowest-performing GroupKFold fold
# ------------------------------------------------------------

# Recreate the exact folds used by GroupKFold.
# Their order matches the order of cv_r2_scores.
group_folds = list(group_cv.split(X_train, y_train, groups=groups_train))

worst_fold_index = int(np.argmin(cv_r2_scores))
worst_fold_r2 = cv_r2_scores[worst_fold_index]

# The validation indices identify the held-out districts in that fold.
_, worst_validation_indices = group_folds[worst_fold_index]

hard_fold_data = train_data.iloc[worst_validation_indices].copy()
hard_districts = sorted(hard_fold_data["Dist Name"].unique())

print("\nLowest-performing GroupKFold fold:")
print(f"Fold number: {worst_fold_index + 1}")
print(f"Fold R²: {worst_fold_r2:.3f}")

print("\nDistricts in this fold:")
for district in hard_districts:
    print("-", district)

# All other training-period districts form the comparison group.
rest_data = train_data.loc[
    ~train_data["Dist Name"].isin(hard_districts)
].copy()

# Comparison uses observations from the 1991-2009 training period only.
characteristics_comparison = pd.DataFrame([
    {
        "Group": "Low-performing fold",
        "Districts": hard_fold_data["Dist Name"].nunique(),
        "Observations": len(hard_fold_data),
        "Average irrigation share": hard_fold_data["irrigation_share"].mean(),
        "Average yield (kg/ha)": hard_fold_data["yield_kg_per_ha"].mean(),
    },
    {
        "Group": "All other training districts",
        "Districts": rest_data["Dist Name"].nunique(),
        "Observations": len(rest_data),
        "Average irrigation share": rest_data["irrigation_share"].mean(),
        "Average yield (kg/ha)": rest_data["yield_kg_per_ha"].mean(),
    },
])

print("\nCharacteristics comparison:")
print(characteristics_comparison.round(3))

# Count districts, rather than repeated annual observations, within each state.
hard_state_distribution = (
    hard_fold_data[["Dist Name", "State Name"]]
    .drop_duplicates()
    ["State Name"]
    .value_counts()
    .rename_axis("State")
    .reset_index(name="Number of districts")
)

rest_state_distribution = (
    rest_data[["Dist Name", "State Name"]]
    .drop_duplicates()
    ["State Name"]
    .value_counts()
    .rename_axis("State")
    .reset_index(name="Number of districts")
)

print("\nState distribution: low-performing fold")
print(hard_state_distribution.to_string(index=False))

print("\nState distribution: all other districts")
print(rest_state_distribution.to_string(index=False))

# Optional: district-level averages for the low-performing fold.
hard_district_profile = (
    hard_fold_data
    .groupby(["State Name", "Dist Name"])
    .agg(
        observations=("Year", "size"),
        average_irrigation_share=("irrigation_share", "mean"),
        average_yield_kg_ha=("yield_kg_per_ha", "mean"),
    )
    .reset_index()
    .sort_values(["State Name", "Dist Name"])
)

print("\nDistrict-level profile: low-performing fold")
print(hard_district_profile.round(3).to_string(index=False))

# ------------------------------------------------------------
# 9. District-level rainfall-shock sensitivity ranking
# ------------------------------------------------------------

# These districts belonged to the lowest-performing GroupKFold fold.
# Interpret their rankings with additional caution.
low_performing_fold_districts = [
    "Aurangabad", "Bhandara", "Boudh", "Cuttack", "Gadchiroli",
    "Gondia", "Jalgaon", "Jharsuguda", "Kalahandi", "Latur",
    "Nasik", "Nawarangpur", "Pune", "Sambalpur", "Solapur",
]

# Calculate each district's historical average feature values using only
# the 1991-2009 training period.
district_profiles = (
    train_data
    .groupby(["Dist Name", "State Name"], as_index=False)[feature_cols]
    .mean()
)

# Scenario 1: normal rainfall, represented by anomaly = 0.
normal_scenario = district_profiles[feature_cols].copy()
normal_scenario["rainfall_anomaly"] = 0

# Scenario 2: one-standard-deviation-below-normal rainfall.
rainfall_shock_scenario = normal_scenario.copy()
rainfall_shock_scenario["rainfall_anomaly"] = -1

# Scenario 3: compound drought: below-normal rainfall and above-normal
# temperature at the same time. All other predictors remain at each
# district's historical average.
compound_shock_scenario = normal_scenario.copy()
compound_shock_scenario["rainfall_anomaly"] = -1
compound_shock_scenario["temp_anomaly"] = 1

# The random forest predicts log yield, so transform predictions to kg/ha.
predicted_yield_normal = np.exp(rf_model.predict(normal_scenario))
predicted_yield_shock = np.exp(rf_model.predict(rainfall_shock_scenario))
predicted_yield_compound = np.exp(
    rf_model.predict(compound_shock_scenario)
)

# Positive values mean a predicted yield loss after the rainfall shock.
predicted_yield_loss_pct = (
    (predicted_yield_normal - predicted_yield_shock)
    / predicted_yield_normal
) * 100

# Positive values mean a larger loss under the compound scenario than under
# the rainfall-only scenario.
predicted_yield_loss_compound_pct = (
    (predicted_yield_normal - predicted_yield_compound)
    / predicted_yield_normal
) * 100

compound_vs_rainfall_only_gap_pct = (
    predicted_yield_loss_compound_pct - predicted_yield_loss_pct
)

# Create the requested district ranking.
climate_sensitivity_ranking = pd.DataFrame({
    "district": district_profiles["Dist Name"],
    "state": district_profiles["State Name"],
    "irrigation_share": district_profiles["irrigation_share"],
    "predicted_yield_loss_pct": predicted_yield_loss_pct,
    "predicted_yield_loss_compound_pct": predicted_yield_loss_compound_pct,
    "compound_vs_rainfall_only_gap_pct": compound_vs_rainfall_only_gap_pct,
    "low_performing_cv_fold": district_profiles["Dist Name"].isin(
        low_performing_fold_districts
    ),
})

climate_sensitivity_ranking = (
    climate_sensitivity_ranking
    .sort_values("predicted_yield_loss_compound_pct", ascending=False)
    .reset_index(drop=True)
)

print("\nDistrict compound drought climate sensitivity ranking:")
print(climate_sensitivity_ranking.round(3).to_string(index=False))

# Overwrite the existing ranking file with the compound-scenario columns.
climate_sensitivity_ranking.to_csv(
    "data/processed/district_climate_sensitivity_ranking.csv",
    index=False
)

print(
    "\nSaved updated ranking to "
    "data/processed/district_climate_sensitivity_ranking.csv"
)

# ------------------------------------------------------------
# State-level rainfall sensitivity summary
# ------------------------------------------------------------
state_sensitivity_summary = (
    climate_sensitivity_ranking
    .groupby("state", as_index=False)
    .agg(
        average_predicted_yield_loss_pct=(
            "predicted_yield_loss_pct", "mean"
        ),
        average_irrigation_share=("irrigation_share", "mean"),
        number_of_districts=("district", "nunique"),
        min_predicted_yield_loss_pct=(
            "predicted_yield_loss_pct", "min"
        ),
        max_predicted_yield_loss_pct=(
            "predicted_yield_loss_pct", "max"
        ),
    )
    .sort_values("average_predicted_yield_loss_pct", ascending=False)
    .reset_index(drop=True)
)

print("\nState-level rainfall sensitivity summary:")
print(state_sensitivity_summary.round(3).to_string(index=False))

# Average additional loss from the compound scenario, by state.
state_compound_gap_summary = (
    climate_sensitivity_ranking
    .groupby("state", as_index=False)
    .agg(
        average_compound_vs_rainfall_only_gap_pct=(
            "compound_vs_rainfall_only_gap_pct", "mean"
        ),
        number_of_districts=("district", "nunique"),
    )
    .sort_values(
        "average_compound_vs_rainfall_only_gap_pct",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nAverage additional loss from compound versus rainfall-only shock:")
print(state_compound_gap_summary.round(3).to_string(index=False))
