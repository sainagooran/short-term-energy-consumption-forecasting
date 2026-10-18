# Short-Term Energy Consumption Forecasting

## Project Overview

This project investigates short-term electricity consumption forecasting using historical hourly energy demand data and machine learning. The main objective is to predict electricity consumption one hour ahead and compare the performance of different forecasting models.

The project uses historical consumption values, calendar-based features, and lag features to capture recent demand patterns and recurring temporal behavior. Three approaches are evaluated: a persistence baseline, Random Forest, and HistGradientBoosting.

The project also includes exploratory data analysis, model evaluation, prediction visualizations, and error analysis to better understand the strengths and limitations of the forecasting approach.

## Objectives

The main objectives of this project are to:

- Explore historical hourly electricity consumption patterns.
- Clean and prepare the dataset for time-series forecasting.
- Engineer calendar-based and lag-based features.
- Predict electricity consumption one hour ahead.
- Compare machine learning models against a simple baseline.
- Evaluate forecasting performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Analyze prediction errors and identify areas for potential improvement.

## Dataset

This project uses the **AEP Hourly Energy Consumption** dataset, publicly available through Kaggle.

The dataset contains historical electricity consumption measurements for the American Electric Power (AEP) region.

| Property | Description |
|---|---|
| Dataset | AEP Hourly Energy Consumption |
| Target variable | `AEP_MW` |
| Original observations | Approximately 121,273 |
| Time period | October 2004 – August 2018 |
| Temporal frequency | Hourly, with some irregular intervals |
| Data format | CSV |

The `AEP_MW` column represents electricity consumption in megawatts (MW).

### Data Preparation

The data preparation process includes:

1. Loading the raw CSV file.
2. Converting timestamps to datetime format.
3. Sorting observations chronologically.
4. Removing duplicate timestamps while retaining the last observation for each duplicated timestamp.
5. Creating calendar-based features, lag features, and the next-hour prediction target.
6. Removing rows with missing required features or target values.

In the reported run, four duplicate timestamps were removed. After feature engineering and the removal of rows with unavailable required values, 120,965 observations remained for model development and evaluation.

Although the dataset is predominantly hourly, some consecutive observations are separated by more than one hour. Therefore, lag features are constructed using timestamp-based lookups rather than assuming that every consecutive row represents exactly one hour.

## Technologies

The project uses the following tools and libraries:

- **Python** — programming language.
- **pandas** — data loading, manipulation, and time-series preparation.
- **NumPy** — numerical operations.
- **Matplotlib** — visualization.
- **scikit-learn** — machine learning models and evaluation metrics.
- **Jupyter Notebook** — exploratory data analysis.

## Project Structure

```text
Short-Term Energy Consumption Forecasting/
│
├── data/
│   └── raw/
│       └── AEP_hourly.csv
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── reports/
│   ├── model_results.csv
│   ├── model_predictions.csv
│   ├── prediction_error_analysis.csv
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── prediction_errors.png
│
├── src/
│   ├── data_loader.py
│   ├── inspect_timestamps.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Main Files

- `data_loader.py`: Loads and prepares the raw dataset.
- `inspect_timestamps.py`: Inspects timestamp continuity and irregular time intervals.
- `train.py`: Creates forecasting features, trains the models, evaluates predictions, and generates reports.
- `01_exploratory_data_analysis.ipynb`: Explores the dataset and visualizes consumption patterns.
- `model_results.csv`: Stores model evaluation metrics.
- `model_predictions.csv`: Stores actual and predicted consumption values.
- `prediction_error_analysis.csv`: Contains additional prediction error analysis.
- PNG files: Visualize model performance and prediction errors.

## Methodology

### 1. Feature Engineering

The forecasting models use historical consumption and time-based information to estimate electricity demand at the next hourly timestamp.

The main features are:

| Feature | Description |
|---|---|
| `AEP_MW` | Consumption at the current timestamp |
| `Hour` | Hour of the day |
| `DayOfWeek` | Day of the week |
| `Month` | Month of the year |
| `Year` | Calendar year |
| `IsWeekend` | Indicates whether the day is a weekend |
| `Lag_1` | Consumption one hour before the current timestamp |
| `Lag_2` | Consumption two hours before the current timestamp |
| `Lag_24` | Consumption 24 hours before the current timestamp |
| `Lag_168` | Consumption 168 hours before the current timestamp |
| `Recent_Change` | Difference between current consumption and `Lag_1` |

Lag features are matched using timestamps. If a required historical timestamp is unavailable, the corresponding feature may be missing and the affected row is excluded from model development.

### 2. Prediction Target

The target variable is electricity consumption at the next exact hourly timestamp:

\[
\text{Target}_t = \text{AEP\_MW}_{t+1\text{ hour}}
\]

This is a **one-hour-ahead forecasting problem**, rather than a multi-day or long-term forecasting task.

### 3. Train-Test Split

The data is divided chronologically, using earlier observations for training and later observations for testing. This preserves the temporal ordering of the data and avoids randomly mixing past and future observations.

In the reported run:

- Training observations: 96,483
- Testing observations: 24,481
- Approximate training proportion: 80%
- Approximate testing proportion: 20%
- Test period starts at: `2015-10-17 05:00:00`

To prevent target leakage across the train-test boundary, training samples whose next-hour target falls within the test period are excluded from training. The last training target is at `2015-10-17 04:00:00`, immediately before the test period begins.

The number of samples remaining after this boundary protection differs slightly from a simple 80/20 split.

## Models

Three forecasting approaches are evaluated.

### 1. Persistence Baseline

The persistence baseline predicts that next-hour electricity consumption will be equal to consumption at the current timestamp.

This simple approach provides a reference point for determining whether machine learning models offer a meaningful improvement.

### 2. Random Forest Regressor

Random Forest is an ensemble learning method that combines multiple decision trees to estimate a continuous target.

Configuration:

- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`

### 3. HistGradientBoosting Regressor

HistGradientBoosting builds an ensemble of decision trees sequentially, with each stage improving upon the errors of the preceding stages.

Configuration:

- `max_iter=300`
- `learning_rate=0.05`
- `max_leaf_nodes=31`
- `random_state=42`

The models use the same engineered features and chronological evaluation framework to support a consistent comparison.

## Evaluation Metrics

Model performance is evaluated using two regression metrics.

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted consumption values.

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

**Root Mean Squared Error (RMSE)** measures the square root of the average squared prediction error. Because larger errors receive a greater penalty, RMSE is particularly useful for identifying models

## Author

Created by **Saina Gooran** as a machine learning project focused on short-term electricity consumption forecasting, time-series feature engineering, and model evaluation.

