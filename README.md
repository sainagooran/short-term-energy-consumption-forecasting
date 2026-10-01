# Short-Term Energy Consumption Forecasting

## Overview

This project investigates short-term electricity consumption forecasting using historical hourly energy consumption data and machine learning techniques.

The main objective is to predict electricity consumption for the following hour based on recent consumption patterns and temporal characteristics such as hour of day, day of week, month, and year.

The project uses a chronological train/test split to preserve the temporal structure of the data and avoid using future observations during model training.

---

## Research Question

**Can machine learning models accurately forecast next-hour electricity consumption using historical consumption patterns and time-based features?**

To investigate this question, several approaches are compared, including a persistence baseline, Linear Regression, Random Forest, and HistGradientBoosting.

---

## Dataset

The project uses the **AEP hourly electricity consumption dataset**.

The dataset contains hourly electricity consumption measurements with the following variables:

| Column     | Description                               |
| ---------- | ----------------------------------------- |
| `Datetime` | Timestamp of the observation              |
| `AEP_MW`   | Electricity consumption in megawatts (MW) |

The dataset contains **121,273 observations** covering multiple years, from October 2004 to August 2018.

Before modeling, the data was sorted chronologically and basic data validation was performed, including checks for missing values, duplicate rows, data types, and the overall time range.

---

## Methodology

The forecasting task is formulated as a **one-step-ahead prediction problem**.

For each observation, the target variable represents electricity consumption at the following time step:

```text
Target(t) = Energy Consumption(t + 1)
```

The input features are based only on information available at the current or previous time steps.

### Feature Engineering

The following temporal features were extracted from the timestamp:

* Hour of day
* Day of week
* Month
* Year

Lag features were also created to capture recent consumption patterns:

* `Lag_1` — previous observation
* `Lag_2` — two observations earlier
* `Lag_24` — 24 observations earlier
* `Lag_168` — 168 observations earlier

The current electricity consumption (`AEP_MW`) was also included as a predictor.

---

## Train/Test Split

Because this is a time-series forecasting problem, a random train/test split was not used.

Instead, the data was divided chronologically:

* **80%** — training data
* **20%** — testing data

This ensures that the models are trained on earlier observations and evaluated on later observations.

### Dataset Split

| Set      | Samples |
| -------- | ------: |
| Training |  96,883 |
| Testing  |  24,221 |

The training data covers observations up to October 2015, while the test set contains later observations through August 2018.

---

## Models

Four approaches were evaluated.

### 1. Persistence Baseline

The current electricity consumption is used directly as the prediction for the next observation.

This provides a simple reference point for evaluating the machine learning models.

### 2. Linear Regression

A Linear Regression model was used as a simple machine learning baseline.

It provides a comparison between a linear model and the nonlinear ensemble models used in this project.

### 3. Random Forest Regressor

Random Forest is an ensemble learning method based on multiple decision trees.

The model was configured with:

* 100 trees
* Fixed random seed (`42`)
* Parallel processing

### 4. HistGradientBoosting Regressor

HistGradientBoosting was used as a second nonlinear ensemble model.

The configuration included:

* 300 boosting iterations
* Learning rate of `0.05`
* Maximum 31 leaf nodes
* Fixed random seed (`42`)

---

## Evaluation Metrics

Two regression metrics were used.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual consumption:

```text
MAE = mean(|y_actual - y_predicted|)
```

Lower values indicate smaller average prediction errors.

### Root Mean Squared Error (RMSE)

RMSE gives greater weight to larger prediction errors:

```text
RMSE = sqrt(mean((y_actual - y_predicted)^2))
```

Lower values indicate smaller prediction errors, with larger errors having a greater influence on the metric.

---

## Results

The models were evaluated on the chronological test set.

| Model                |   MAE (MW) |  RMSE (MW) |
| -------------------- | ---------: | ---------: |
| Persistence Baseline |     407.72 |     525.66 |
| Linear Regression    |     195.05 |     267.26 |
| Random Forest        |     129.79 |     179.28 |
| HistGradientBoosting | **126.44** | **169.65** |

Among the evaluated models, HistGradientBoosting produced the lowest MAE and RMSE on the test set.

The results also show a substantial reduction in prediction error compared with the persistence baseline.

---

## Feature Importance

Feature importance was examined using the Random Forest model.

The most influential feature was the current electricity consumption (`AEP_MW`), with an importance value of approximately **0.956**.

The `Hour` feature was the second most influential feature, reflecting daily consumption patterns.

Other lag and temporal features contributed smaller amounts of additional information.

---

## Visualizations

The project includes visual analysis of:

* Historical electricity consumption
* Random Forest feature importance
* Actual versus predicted electricity consumption

The visualizations are included in the Jupyter Notebook:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

---

## Project Structure

```text
Short-Term Energy Consumption Forecasting/

├── data/
│   └── raw/
│       └── AEP_hourly.csv
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies

* Python 3.12
* pandas
* NumPy
* Matplotlib
* scikit-learn
* Jupyter Notebook

---

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run the training pipeline

```bash
python src/train.py
```

The script trains the Random Forest and HistGradientBoosting models and reports their MAE and RMSE values on the test set.

### Explore the analysis

Open:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

The notebook contains the exploratory analysis, feature engineering, model evaluation, and visualizations.

---

## Limitations

Several limitations should be considered.

First, the lag features are based on observation positions rather than explicitly matching calendar timestamps. Therefore, `Lag_24` and `Lag_168` represent lagged observations and do not necessarily correspond to exactly 24 or 168 hours in every case.

Second, the models were evaluated using a single chronological train/test split. Although this preserves temporal ordering, additional time-series cross-validation could provide a more robust estimate of model performance.

Finally, the project focuses on relatively simple machine learning approaches and does not include external variables such as weather, holidays, or electricity prices.

---

## Future Work

Possible extensions include:

* More robust handling of irregular timestamps
* Additional time-series features
* Holiday and seasonal indicators
* Weather-related features
* Time-series cross-validation
* Hyperparameter tuning
* Comparison with dedicated forecasting models
* Evaluation over different forecasting horizons

---

## Conclusion

This project demonstrates a machine learning approach to short-term electricity consumption forecasting using historical hourly data.

The results show that the evaluated ensemble models captured useful relationships between recent electricity consumption and temporal patterns. HistGradientBoosting achieved the lowest MAE and RMSE among the models evaluated on the chronological test set.

The project also demonstrates a complete machine learning workflow, including data validation, temporal feature engineering, time-aware model evaluation, model comparison, and reproducible model training.
