
from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"


def load_data():
    """Load the energy consumption dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Check if required columns exist
    required_columns = ["Datetime", "AEP_MW"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Convert columns to the correct data types
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["AEP_MW"] = pd.to_numeric(df["AEP_MW"], errors="coerce")

    return df


def validate_data(df):
    """Check the basic quality of the dataset."""

    print("\n--- Dataset Information ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Invalid Datetimes ---")
    print(df["Datetime"].isnull().sum())

    print("\n--- Duplicate Timestamps ---")
    print(df["Datetime"].dropna().duplicated().sum())

    print("\n--- Time Range ---")
    print(f"Start: {df['Datetime'].min()}")
    print(f"End:   {df['Datetime'].max()}")

    print("\n--- Energy Consumption Statistics ---")
    print(df["AEP_MW"].describe())




def check_time_intervals(df):
    """Check the time intervals between consecutive readings."""

    # Sort timestamps before checking the intervals
    timestamps = (
        df["Datetime"]
        .dropna()
        .sort_values()
        .reset_index(drop=True)
    )

    # Calculate the time difference between consecutive readings
    time_diff = timestamps.diff()

    # Show duplicate timestamps
    print("\n--- Duplicate Timestamps ---")
    print(timestamps[timestamps.duplicated(keep=False)])

    # Inspect energy consumption at duplicate timestamps
    print("\n--- Consumption at Duplicate Timestamps ---")

    duplicate_rows = df[
        df["Datetime"].duplicated(keep=False)
    ].sort_values("Datetime")

    print(duplicate_rows)

    # Show intervals longer than one hour
    print("\n--- Irregular Time Intervals ---")

    for i in range(1, len(timestamps)):
        if time_diff.iloc[i] > pd.Timedelta(hours=1):
            print(
                f"Previous: {timestamps.iloc[i - 1]} | "
                f"Current: {timestamps.iloc[i]} | "
                f"Gap: {time_diff.iloc[i]}"
            )

    # Inspect selected time intervals
    print("\n--- Checking Unexpected Gaps ---")

    dates_to_check = [
        "2010-12-09 23:00:00",
        "2012-12-06 03:00:00",
        "2014-03-11 13:00:00",
    ]

    for date in dates_to_check:
        timestamp = pd.Timestamp(date)

        nearby_rows = df[
            df["Datetime"].between(
                timestamp - pd.Timedelta(hours=1),
                timestamp + pd.Timedelta(hours=2),
            )
        ].sort_values("Datetime")

        print(f"\nAround {timestamp}:")
        print(nearby_rows)

    # Summarize time intervals
    print("\n--- Time Interval Summary ---")
    print(f"Total intervals checked: {time_diff.dropna().shape[0]}")
    print(
        "Intervals longer than 1 hour:",
        (time_diff > pd.Timedelta(hours=1)).sum()
    )
    print(
        "Intervals shorter than 1 hour:",
        (time_diff < pd.Timedelta(hours=1)).sum()
    )


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")

    print("\n--- First 5 Rows ---")
    print(df.head())

    validate_data(df)
    check_time_intervals(df)
