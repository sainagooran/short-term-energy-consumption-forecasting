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

- `src/data_loader.py`: Loads and prepares the raw dataset.
- `src/inspect_timestamps.py`: Inspects timestamp continuity and irregular time intervals.
- `src/train.py`: Creates forecasting features, trains the models, evaluates predictions, and generates reports.
- `notebooks/01_exploratory_data_analysis.ipynb`: Explores the dataset and visualizes consumption patterns.
- `reports/model_results.csv`: Stores model evaluation metrics.
- `reports/model_predictions.csv`: Stores actual and predicted consumption values.
- `reports/prediction_error_analysis.csv`: Contains additional prediction error analysis.
- `reports/model_comparison.png`: Compares model evaluation metrics.
- `reports/actual_vs_predicted.png`: Compares actual and predicted consumption.
- `reports/prediction_errors.png`: Visualizes prediction errors.
- `requirements.txt`: Lists the Python dependencies.
- `.gitignore`: Specifies files and directories Git should ignore.

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

**Root Mean Squared Error (RMSE)** measures the square root of the average squared prediction error. Because larger errors receive a greater penalty, RMSE is particularly useful for identifying models that occasionally make substantial forecasting errors.

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

For both metrics, lower values indicate better predictive performance.

## Results

The following results are from the reported model run.

| Model | MAE (MW) | RMSE (MW) |
|---|---:|---:|
| Persistence Baseline | 407.17 | 525.16 |
| Random Forest | 129.60 | 178.33 |
| HistGradientBoosting | 118.03 | 158.72 |

HistGradientBoosting achieved the lowest MAE and RMSE among the three evaluated approaches.

Compared with the persistence baseline, HistGradientBoosting reduced MAE by approximately **71%**. This indicates a substantial improvement in average absolute prediction accuracy on the evaluated test set.

Random Forest also performed considerably better than the persistence baseline, although its errors were higher than those of HistGradientBoosting in this experiment.

These results apply to the dataset, features, model configurations, and evaluation procedure used in this project. They should not be interpreted as a guarantee of performance on other regions or future operating conditions.

## Visualizations

The following figures are generated by the training script and saved in the `reports/` directory.

### 1. Model Comparison

This figure compares the evaluation metrics of the Persistence Baseline, Random Forest, and HistGradientBoosting models.

![Model Comparison](reports/model_comparison.png)

### 2. Actual vs. Predicted Consumption

This figure compares actual electricity consumption with predicted values over the test period.

![Actual vs. Predicted Consumption](reports/actual_vs_predicted.png)

### 3. Prediction Errors

This figure visualizes forecasting errors across the test period and helps identify instances where predictions differ substantially from actual consumption.

![Prediction Errors](reports/prediction_errors.png)

## Error Analysis

In addition to overall MAE and RMSE, the project examines the distribution of prediction errors and model performance during periods of changing electricity demand.

In the reported run:

- The first seven test days had a mean error of approximately **-6.80 MW** and an MAE of **120.16 MW**.
- Across the full test set, the mean error was approximately **-17.17 MW** and the MAE was **118.03 MW**.
- The largest absolute prediction error was approximately **3,680.11 MW**.
- The largest 5% of absolute hourly consumption changes had an MAE of approximately **157.05 MW**, compared with **115.98 MW** for the remaining changes.

Prediction error is defined as actual consumption minus predicted consumption. The negative mean error indicates a slight average tendency toward overprediction.

The higher MAE during large consumption changes suggests that sudden demand fluctuations are more difficult to predict using the current feature set.

Individual large errors should be investigated alongside their timestamps, historical observations, and data quality before being attributed to a specific cause.

## Generated Outputs

Running the training script generates evaluation and analysis files in the `reports/` directory:

- `model_results.csv`
- `model_predictions.csv`
- `prediction_error_analysis.csv`
- `model_comparison.png`
- `actual_vs_predicted.png`
- `prediction_errors.png`

These outputs can be used to compare models, inspect individual forecasts, and support further analysis.

## Installation and Usage

### Requirements

- Python 3.10 or a compatible Python version supported by the installed dependencies.
- pip
- The dataset file `AEP_hourly.csv`

### 1. Clone the Repository

```bash
git clone https://github.com/sainagooran/short-term-energy-consumption-forecasting.git
cd short-term-energy-consumption-forecasting
```

### 2. Create a Virtual Environment

On Windows, using PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place `AEP_hourly.csv` in the following location:

```text
data/raw/AEP_hourly.csv
```

### 5. Run the Training Script

From the project root directory, run:

```bash
python src/train.py
```

The script loads the dataset, prepares the forecasting features, trains the models, calculates evaluation metrics, and generates the configured reports and visualizations.

### 6. Explore the Results

Review the files in `reports/` to compare model performance and examine predictions and errors.

The exploratory analysis notebook can also be opened in Jupyter Notebook or Visual Studio Code.

## Limitations

This project has several limitations:

- The analysis focuses on a single regional electricity consumption series.
- Only a limited set of historical and calendar-based features is used.
- Weather conditions, holidays, electricity prices, and other external variables are not included.
- Irregular timestamp intervals may affect the availability of lag features.
- The persistence baseline and two tree-based models do not represent every possible forecasting approach.
- Aggregate metrics may hide periods of particularly poor performance.
- Performance may change during unusual demand patterns or abrupt consumption changes.

This project is an educational forecasting study and is not intended to serve as a production electricity demand forecasting system.

## Future Improvements

Potential extensions include:

- Evaluating seasonal-naive baselines based on consumption 24 and 168 hours earlier.
- Investigating irregular timestamps and daylight-saving-time transitions in greater detail.
- Adding weather, holiday, and other relevant external features.
- Comparing additional models, such as linear regression and time-series-specific forecasting methods.
- Performing hyperparameter tuning using a validation period separate from the final test set.
- Evaluating performance across different seasons and demand conditions.
- Improving uncertainty estimation and examining forecast performance during demand peaks.

## Conclusion

This project demonstrates a practical workflow for short-term electricity consumption forecasting, from data preparation and feature engineering to model comparison and error analysis.

In the reported experiment, both Random Forest and HistGradientBoosting outperformed the persistence baseline, with HistGradientBoosting achieving the lowest MAE and RMSE.

The findings suggest that historical consumption and calendar-based features provide useful information for one-hour-ahead forecasting. However, prediction errors increase during substantial consumption changes, highlighting an opportunity for further improvement.

Further feature engineering, validation, and evaluation of alternative forecasting approaches could help establish the robustness of these findings.

## Author

**Saina Gooran**

Created as a machine learning project focused on short-term electricity consumption forecasting, time-series feature engineering, model evaluation, and error analysis.


