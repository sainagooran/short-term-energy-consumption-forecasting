
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load dataset
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "AEP_hourly.csv"

df = pd.read_csv(data_path)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)


# Create time-based features
df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
df["Month"] = df["Datetime"].dt.month
df["Year"] = df["Datetime"].dt.year


# Create lag features
df["Lag_1"] = df["AEP_MW"].shift(1)
df["Lag_2"] = df["AEP_MW"].shift(2)
df["Lag_24"] = df["AEP_MW"].shift(24)
df["Lag_168"] = df["AEP_MW"].shift(168)


# Target: energy consumption in the next hour
df["Target"] = df["AEP_MW"].shift(-1)

df = df.dropna().copy()


# Features and target
features = [
    "AEP_MW",
    "Hour",
    "DayOfWeek",
    "Month",
    "Year",
    "Lag_1",
    "Lag_2",
    "Lag_24",
    "Lag_168"
]

X = df[features]
y = df["Target"]


# Time-based train/test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))


# Gradient Boosting
gb_model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))


# Results
print("\n--- Results ---")

print("\nRandom Forest")
print(f"MAE:  {rf_mae:.2f} MW")
print(f"RMSE: {rf_rmse:.2f} MW")

print("\nHistGradientBoosting")
print(f"MAE:  {gb_mae:.2f} MW")
print(f"RMSE: {gb_rmse:.2f} MW")

