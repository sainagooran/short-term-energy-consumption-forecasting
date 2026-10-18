
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data


# --------------------------------------------------
# 1. Load and prepare dataset
# --------------------------------------------------

df = load_data()
df = df.sort_values("Datetime").reset_index(drop=True)

# Keep the last reading at each duplicated timestamp
df = df.drop_duplicates(subset="Datetime", keep="last").copy()
df = df.reset_index(drop=True)


# --------------------------------------------------
# 2. Check time intervals
# --------------------------------------------------

time_diffs = df["Datetime"].diff()

print("\n--- Time Interval Analysis ---")
print("Time interval counts:")

print(
    time_diffs.value_counts()
    .sort_index()
    .to_string()
)

non_hourly = time_diffs[
    time_diffs.notna()
    & (time_diffs != pd.Timedelta(hours=1))
]

print(f"\nNon-hourly intervals: {len(non_hourly)}")

if not non_hourly.empty:
    print("\nExamples of non-hourly intervals:")

    interval_examples = df.loc[
        non_hourly.index, ["Datetime"]
    ].copy()

    interval_examples["Time_Difference"] = time_diffs.loc[
        non_hourly.index
    ]

    print(interval_examples.head(20).to_string(index=False))


# --------------------------------------------------
# 3. Create time-based features
# --------------------------------------------------

df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
df["Month"] = df["Datetime"].dt.month
df["Year"] = df["Datetime"].dt.year
df["IsWeekend"] = (
    df["Datetime"].dt.dayofweek >= 5
).astype(int)


# --------------------------------------------------
# 4. Create lag features and target

df = df.set_index("Datetime").sort_index()

df["Lag_1"] = df["AEP_MW"].reindex(
    df.index - pd.Timedelta(hours=1)
).to_numpy()

df["Lag_2"] = df["AEP_MW"].reindex(
    df.index - pd.Timedelta(hours=2)
).to_numpy()

df["Lag_24"] = df["AEP_MW"].reindex(
    df.index - pd.Timedelta(hours=24)
).to_numpy()

df["Lag_168"] = df["AEP_MW"].reindex(
    df.index - pd.Timedelta(hours=168)
).to_numpy()

df["Recent_Change"] = df["AEP_MW"] - df["Lag_1"]

# Target: consumption exactly one hour later
df["Target"] = df["AEP_MW"].reindex(
    df.index + pd.Timedelta(hours=1)
).to_numpy()

# Restore Datetime as a regular column
df = df.reset_index()



# --------------------------------------------------
# 5. Check and remove missing values
# --------------------------------------------------

required_columns = [
    "AEP_MW",
    "Hour",
    "DayOfWeek",
    "Month",
    "Year",
    "IsWeekend",
    "Lag_1",
    "Lag_2",
    "Lag_24",
    "Lag_168",
    "Recent_Change",
    "Target",
]

missing_counts = df[required_columns].isna().sum()

print("\n--- Missing Values Before Dropping Rows ---")
print(missing_counts[missing_counts > 0].to_string())

rows_before = len(df)

df = df.dropna(subset=required_columns).copy()

rows_after = len(df)
rows_removed = rows_before - rows_after

print(f"\nRows before dropna: {rows_before}")
print(f"Rows removed: {rows_removed}")
print(f"Rows remaining: {rows_after}")


# --------------------------------------------------
# 6. Define features and target
# --------------------------------------------------

features = [
    "AEP_MW",
    "Hour",
    "DayOfWeek",
    "Month",
    "Year",
    "IsWeekend",
    "Lag_1",
    "Lag_2",
    "Lag_24",
    "Lag_168",
    "Recent_Change",
]

X = df[features]
y = df["Target"]


# --------------------------------------------------
# 7. Time-based train/test split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 8. Persistence baseline
# --------------------------------------------------

# Predict the next row using the current consumption
baseline_predictions = X_test["AEP_MW"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)


# --------------------------------------------------
# 9. Random Forest
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)


# --------------------------------------------------
# 10. HistGradientBoosting
# --------------------------------------------------

gb_model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42,
)

gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = np.sqrt(
    mean_squared_error(y_test, gb_predictions)
)


# --------------------------------------------------
# 11. Compare model results
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Persistence Baseline",
        "Random Forest",
        "HistGradientBoosting",
    ],
    "MAE": [
        baseline_mae,
        rf_mae,
        gb_mae,
    ],
    "RMSE": [
        baseline_rmse,
        rf_rmse,
        gb_rmse,
    ],
})

print("\n--- Results ---")
print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# --------------------------------------------------
# 12. Set output directory and save results
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

results_path = reports_dir / "model_results.csv"
results.to_csv(results_path, index=False)

print(f"\nResults saved to: {results_path}")


# --------------------------------------------------
# 13. Plot model comparison
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

results.plot(
    x="Model",
    y="MAE",
    kind="bar",
    ax=axes[0],
    legend=False,
    title="Model Comparison - MAE",
)

results.plot(
    x="Model",
    y="RMSE",
    kind="bar",
    ax=axes[1],
    legend=False,
    title="Model Comparison - RMSE",
)

axes[0].set_ylabel("MAE (MW)")
axes[1].set_ylabel("RMSE (MW)")

for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()

comparison_path = reports_dir / "model_comparison.png"
plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Model comparison plot saved to: {comparison_path}")


# --------------------------------------------------
# 14. Plot actual vs predicted consumption
# --------------------------------------------------

plot_points = min(24 * 7, len(y_test))

actual_values = y_test.iloc[:plot_points].to_numpy()
predicted_values = gb_predictions[:plot_points]

test_dates = df["Datetime"].iloc[
    split_index:split_index + plot_points
]

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    test_dates,
    actual_values,
    label="Actual Consumption",
    linewidth=2,
)

ax.plot(
    test_dates,
    predicted_values,
    label="HistGradientBoosting Prediction",
    linewidth=2,
)

ax.set_title("Actual vs Predicted Electricity Consumption")
ax.set_xlabel("Datetime")
ax.set_ylabel("Electricity Consumption (MW)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()

prediction_plot_path = reports_dir / "actual_vs_predicted.png"
plt.savefig(prediction_plot_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Actual vs predicted plot saved to: {prediction_plot_path}")


# --------------------------------------------------
# 15. Analyze errors in the first 7 test days
# --------------------------------------------------

prediction_errors = actual_values - predicted_values

print("\n--- Prediction Error Analysis (First 7 Test Days) ---")
print(f"Mean Error: {np.mean(prediction_errors):.2f} MW")
print(
    f"Mean Absolute Error: "
    f"{np.mean(np.abs(prediction_errors)):.2f} MW"
)
print(
    f"Maximum Absolute Error: "
    f"{np.max(np.abs(prediction_errors)):.2f} MW"
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    test_dates,
    prediction_errors,
    label="Actual - Predicted",
)

ax.axhline(y=0, linestyle="--", linewidth=1)
ax.set_title("HistGradientBoosting Prediction Errors")
ax.set_xlabel("Datetime")
ax.set_ylabel("Error (MW)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()

error_plot_path = reports_dir / "prediction_errors.png"
plt.savefig(error_plot_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Prediction error plot saved to: {error_plot_path}")


# --------------------------------------------------
# 16. Analyze errors on the full test set
# --------------------------------------------------

all_errors = y_test.to_numpy() - gb_predictions

print("\n--- Full Test Set Error Analysis ---")
print(f"Mean Error: {np.mean(all_errors):.2f} MW")
print(
    f"Mean Absolute Error: "
    f"{np.mean(np.abs(all_errors)):.2f} MW"
)
print(
    f"Maximum Absolute Error: "
    f"{np.max(np.abs(all_errors)):.2f} MW"
)
print(f"Minimum Error: {np.min(all_errors):.2f} MW")
print(f"Maximum Error: {np.max(all_errors):.2f} MW")


# --------------------------------------------------
# 17. Save and display largest prediction errors
# --------------------------------------------------

error_analysis = pd.DataFrame({
    "Datetime": df["Datetime"].iloc[split_index:].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Predicted": gb_predictions,
    "Error": all_errors,
})

error_analysis["Absolute_Error"] = error_analysis["Error"].abs()

worst_predictions = error_analysis.nlargest(
    10, "Absolute_Error"
)

print("\n--- 10 Largest Prediction Errors ---")
print(
    worst_predictions[
        [
            "Datetime",
            "Actual",
            "Predicted",
            "Error",
            "Absolute_Error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

error_path = reports_dir / "prediction_error_analysis.csv"
error_analysis.to_csv(error_path, index=False)

# Save model predictions
predictions_path = reports_dir / "model_predictions.csv"

predictions_df = pd.DataFrame({
    "Datetime": df["Datetime"].iloc[split_index:].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Persistence_Baseline": baseline_predictions.to_numpy(),
    "Random_Forest": rf_predictions,
    "HistGradientBoosting": gb_predictions,
})

predictions_df.to_csv(predictions_path, index=False)

print(f"Model predictions saved to: {predictions_path}")

print(f"\nError analysis saved to: {error_path}")


# --------------------------------------------------
# 18. Analyze sudden changes in actual consumption
# --------------------------------------------------

error_analysis["Actual_Change"] = (
    error_analysis["Actual"].diff().abs()
)

print("\n--- Largest Actual Consumption Changes ---")

largest_changes = error_analysis.nlargest(
    10, "Actual_Change"
)

print(
    largest_changes[
        [
            "Datetime",
            "Actual",
            "Predicted",
            "Actual_Change",
            "Absolute_Error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

# Compare prediction errors during large changes
change_threshold = error_analysis["Actual_Change"].quantile(0.95)

large_change_rows = error_analysis[
    error_analysis["Actual_Change"] >= change_threshold
]

normal_change_rows = error_analysis[
    error_analysis["Actual_Change"] < change_threshold
]

print("\n--- Error Comparison ---")
print(f"95th percentile change threshold: {change_threshold:.2f} MW")

if not large_change_rows.empty and not normal_change_rows.empty:
    print(
        "MAE during largest 5% changes: "
        f"{large_change_rows['Absolute_Error'].mean():.2f} MW"
    )
    print(
        "MAE during other changes: "
        f"{normal_change_rows['Absolute_Error'].mean():.2f} MW"
    )


# --------------------------------------------------
# 18. Inspect data around the largest known error
# --------------------------------------------------

problem_time = pd.Timestamp("2015-12-27 14:00:00")

window_start = problem_time - pd.Timedelta(hours=5)
window_end = problem_time + pd.Timedelta(hours=5)

nearby_data = df.loc[
    df["Datetime"].between(window_start, window_end),
    [
        "Datetime",
        "AEP_MW",
        "Lag_1",
        "Lag_24",
        "Lag_168",
        "Target",
    ],
].copy()

print("\n--- Data Around the Largest Prediction Error ---")
print(
    nearby_data.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)
